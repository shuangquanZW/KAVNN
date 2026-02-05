import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch import Tensor, nn
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import Metric
from torchmetrics import AUROC, Accuracy, Precision, Recall, F1Score, HammingDistance

import pytorch_lightning as pl
from transformers import get_linear_schedule_with_warmup  # 用于学习率预热
from shap import DeepExplainer, GradientExplainer

from dl.BINN.BINNLayer import BINNLayer

from dl.util_dl import (
    load_data,
    create_tensor,
    move_device,
)


def select_tensor(data: np.ndarray, index: list[int]) -> Tensor:
    return torch.tensor(data[index], dtype=torch.float32)


def move_tensor(data: np.ndarray, dtype: torch.dtype = torch.float32):
    return torch.tensor(data, dtype=dtype)


class SubsetAccuracy(Metric):

    def __init__(self, num_labels: int, threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.num_labels = num_labels  # 标签数量
        self.threshold = threshold  # 概率阈值

    def update(self, preds: Tensor, target: Tensor) -> None:
        if preds.dtype == torch.float32:
            preds = (preds >= self.threshold).float()
        sample_matches = torch.all(preds == target, dim=1)
        self.correct += sample_matches.sum()
        self.total += target.shape[0]

    def compute(self) -> Tensor:
        return self.correct.float() / self.total  # type: ignore


class BINNDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        random_state: int = 42,
        organ: str | None = None,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_state = random_state
        self.organ = organ

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str | None = None) -> None:
        if self.organ == "liver":
            index = np.load(self.data_dir + "/edge_index/liver_index.npy")
        elif self.organ == "kidney":
            index = np.load(self.data_dir + "/edge_index/kidney_index.npy")
        else:
            index = np.arange(5750)
        # load
        expr = np.load(self.data_dir + "/expr.npy")[index]
        compound = np.load(self.data_dir + "/compound.npy")[index]
        label = np.load(self.data_dir + "/label.npy")[index]
        # split
        length = label.shape[0]
        train_index, no_train = train_test_split(
            np.arange(length),
            test_size=0.4,
            random_state=self.random_state,
        )
        valid_index, test_index = train_test_split(
            no_train,
            test_size=0.5,
            random_state=self.random_state,
        )
        # predict
        if self.organ == "liver":
            expr_pred = np.load(self.data_dir + "/drug_matrix_liver.npy")
            comp_pred = np.load(self.data_dir + "/drug_matrix_liver_compound.npy")
            label_pred = np.load(self.data_dir + "/drug_matrix_liver_label.npy")
        elif self.organ == "kidney":
            expr_pred = np.load(self.data_dir + "/drug_matrix_kidney.npy")
            comp_pred = np.load(self.data_dir + "/drug_matrix_kidney_compound.npy")
            label_pred = np.load(self.data_dir + "/drug_matrix_kidney_label.npy")
        else:
            expr_pred = None
            comp_pred = None
            label_pred = None
        # 保存数据
        if stage == "fit" or stage is None:
            self.train_dataset = TensorDataset(
                select_tensor(expr, train_index),
                select_tensor(compound, train_index),
                select_tensor(label, train_index),
            )
            self.valid_dataset = TensorDataset(
                select_tensor(expr, valid_index),
                select_tensor(compound, valid_index),
                select_tensor(label, valid_index),
            )
        if stage == "test" or stage is None:
            self.test_dataset = TensorDataset(
                select_tensor(expr, test_index),
                select_tensor(compound, test_index),
                select_tensor(label, test_index),
            )
            if expr_pred is not None:
                self.predict_dataset = TensorDataset(
                    move_tensor(expr_pred),
                    move_tensor(comp_pred),  # type: ignore
                    move_tensor(label_pred),  # type: ignore
                )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class BINN(pl.LightningModule):

    def __init__(
        self,
        num_gene_go: int,
        num_go: int,
        num_go_ke: int,
        num_ke: int,
        num_ke_ke: int,
        num_neurals: int = 8,
        aggregation: str = "mean",
        lr: float = 1e-3,
        num_labels: int = 1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.binn = BINNLayer(
            num_gene_go, num_go, num_go_ke, num_ke, num_ke_ke, num_neurals, num_labels
        )
        # 记录训练步数，用于学习率预热
        self.total_steps = 0
        # 损失函数
        self.loss_function = nn.BCEWithLogitsLoss(reduction=aggregation)
        # 评估器
        self.ham_evaluator = HammingDistance("binary")
        self.auc_evaluator = AUROC("binary")
        self.acc_evaluator = Accuracy("binary")
        self.precision_evaluator = Precision("binary")
        self.recall_evaluator = Recall("binary")
        self.f1_evaluator = F1Score("binary")

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            # 加载边索引文件
            gene_go = np.load("data/edge_index/gene_go_bp.npy")
            go_ke = np.load("data/edge_index/go_ke_bp.npy")
            ke_ke = np.load("data/edge_index/ke_ke.npy")
            tissue = np.load("data/edge_index/tissue.npy")
            # 转化为tensor
            gene_go = torch.tensor(gene_go, dtype=torch.long, device=self.device)
            go_ke = torch.tensor(go_ke, dtype=torch.long, device=self.device)
            ke_ke = torch.tensor(ke_ke, dtype=torch.long, device=self.device)
            tissue = torch.tensor(tissue, dtype=torch.long, device=self.device)
            # 保存数据
            self.edge_indices = gene_go, go_ke, ke_ke, tissue

    def forward(
        self,
        gene: Tensor,
        gene_go: Tensor,
        go_ke: Tensor,
        ke_ke: Tensor,
        tissue: Tensor,
        compound: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return self.binn(gene, gene_go, go_ke, ke_ke, tissue, compound)

    def training_step(self, batch: Tensor) -> Tensor:
        gene, compound_state, label = batch
        pred = self(gene, *self.edge_indices, compound_state)
        loss = self.loss_function(pred, label)
        self.log(
            "train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )
        # 记录训练步数，用于学习率预热
        self.total_steps += 1
        return loss

    def validation_step(self, batch: Tensor) -> None:
        gene, compound_state, label = batch
        pred = self(gene, *self.edge_indices, compound_state)
        loss = self.loss_function(pred, label)
        label = label.long()
        ham = self.ham_evaluator(pred, label)
        self.log("val_loss", loss, prog_bar=True, logger=True, on_epoch=True)
        self.log("val_ham", ham, prog_bar=True, logger=True, on_epoch=True)

    def test_step(self, batch: Tensor) -> dict[str, float]:
        gene, compound_state, label = batch
        pred = self(gene, *self.edge_indices, compound_state)
        label = label.long()
        ham = self.ham_evaluator(pred, label)
        auc = self.auc_evaluator(pred, label)
        acc = self.acc_evaluator(pred, label)
        precision = self.precision_evaluator(pred, label)
        recall = self.recall_evaluator(pred, label)
        f1 = self.f1_evaluator(pred, label)
        self.log("test_ham", ham, prog_bar=True, logger=True, on_epoch=True)
        self.log("test_auc", auc, prog_bar=True, logger=True, on_epoch=True)
        self.log("test_acc", acc, prog_bar=True, logger=True, on_epoch=True)
        self.log("test_precision", precision, prog_bar=True, logger=True, on_epoch=True)
        self.log("test_recall", recall, prog_bar=True, logger=True, on_epoch=True)
        self.log("test_f1", f1, prog_bar=True, logger=True, on_epoch=True)
        # 返回当前步的指标
        return {
            "ham": ham.item(),
            "auc": auc.item(),
            "acc": acc.item(),
            "precision": precision.item(),
            "recall": recall.item(),
            "f1": f1.item(),
        }

    def predict_step(self, batch: Tensor) -> Tensor:
        gene, compound_state, _ = batch
        pred = self(gene, *self.edge_indices, compound_state)
        return pred

    def configure_optimizers(self):  # type: ignore
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"])
        scheduler_configs = {}
        # 计算总训练步数
        total_training_steps = self.trainer.estimated_stepping_batches
        warmup_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=10,
            num_training_steps=total_training_steps,
        )
        scheduler_configs = {
            "scheduler": warmup_scheduler,
            "interval": "step",
            "frequency": 1,
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_configs,
        }

    def get_background_data(self, batch: str) -> Tensor:
        metadata = pd.read_csv("./data/source/metadata.csv")
        metadata_batch = metadata[metadata["batch"] == int(batch)]
        num_class = int(metadata_batch["class"].tolist()[0])
        class_index = metadata[metadata["class"] == num_class].index.tolist()
        expr = torch.from_numpy(np.load("./data/expr.npy")).to(self.device)
        background_data = expr[class_index]
        return background_data

    def get_mask_gene_index(self, drug_batch: str) -> list[int]:
        df = pd.read_csv(f"./volcano/{drug_batch}.csv")
        mask_gene_index = []
        for idx, row in df.iterrows():
            if row["log2FC"] >= 0.5 and row["FDR"] <= 0.05:
                mask_gene_index.append(idx)
        return mask_gene_index

    def get_mask(
        self, drug_batch: str
    ) -> tuple[dict[str, float], dict[str, float]] | None:
        import random

        drug, batch = drug_batch.split("_")
        background_data = self.get_background_data(batch)
        mask_gene_index = self.get_mask_gene_index(drug_batch)
        # preprocess data
        expr, compound, label = load_data()
        expr, label, compound = create_tensor(expr, label, compound, device=self.device)
        # get test data
        metadata = pd.read_csv("./data/source/metadata.csv")
        metadata = metadata[
            (metadata["dose"] != "control") & (metadata["compound"] == drug)
        ]
        index = metadata["index"].tolist()
        if not index:
            print(f"No test data found for drug & batch: {drug_batch}")
            return None
        expr = expr[index]
        label = label[index].long()
        compound = compound[index]
        # get mask
        random_index = random.choices(range(background_data.shape[0]), k=len(index))
        background_data = background_data[random_index]
        baseline = expr.clone()
        baseline[:, mask_gene_index] = background_data[:, mask_gene_index]
        # model
        model = self.binn.eval().to(self.device)
        self.edge_indices = move_device(*self.edge_indices, device=self.device)
        # get result
        baseline_result = model(baseline, *self.edge_indices, compound)
        test_result = model(expr, *self.edge_indices, compound)
        # evaluate
        auc_baseline = self.auc_evaluator(baseline_result, label).item()
        acc_baseline = self.acc_evaluator(baseline_result, label).item()
        precision_baseline = self.precision_evaluator(baseline_result, label).item()
        recall_baseline = self.recall_evaluator(baseline_result, label).item()
        f1_baseline = self.f1_evaluator(baseline_result, label).item()
        auc_test = self.auc_evaluator(test_result, label).item()
        acc_test = self.acc_evaluator(test_result, label).item()
        precision_test = self.precision_evaluator(test_result, label).item()
        recall_test = self.recall_evaluator(test_result, label).item()
        f1_test = self.f1_evaluator(test_result, label).item()
        baseline_dict = {
            "auc": auc_baseline,
            "acc": acc_baseline,
            "precision": precision_baseline,
            "recall": recall_baseline,
            "f1": f1_baseline,
        }
        test_dict = {
            "auc": auc_test,
            "acc": acc_test,
            "precision": precision_test,
            "recall": recall_test,
            "f1": f1_test,
        }
        return baseline_dict, test_dict

    def node_explain(
        self, drug_batch: str, method: str = "shap"
    ) -> dict[str, np.ndarray] | None:
        drug, batch = drug_batch.split("_")
        background_data = self.get_background_data(batch)
        # preprocess data
        expr, compound, _ = load_data()
        expr, compound = create_tensor(expr, compound, device=self.device)
        # get test data
        metadata = pd.read_csv("./data/source/metadata.csv")
        metadata = metadata[
            (metadata["dose"] != "control") & (metadata["compound"] == drug)
        ]
        index = metadata["index"].tolist()
        if not index:
            print(f"No test data found for drug & batch: {drug_batch}")
            return None
        test_data = expr[index]
        compound = compound[index]
        test_data = torch.cat((test_data, compound), dim=-1)
        background_data = torch.cat(
            (
                background_data,
                compound[0].unsqueeze(0).expand(background_data.shape[0], -1),
            ),
            dim=-1,
        )
        # explain
        target_layers = ["gene", "go", "ke"]
        result = {}
        if method == "shap":
            for layer_name in target_layers:
                print(f"Computing SHAP values for layer: {layer_name}")
                model = BINNExplainer(
                    self.binn, self.edge_indices, layer_name, self.device
                )
                baseline = model.get_tensor(background_data)
                inputs = model.get_tensor(test_data)
                explainer = DeepExplainer(model, baseline)
                shap_values = explainer.shap_values(inputs, check_additivity=False)
                shap_values = np.array(shap_values)
                if shap_values.ndim == 3:
                    shap_values = shap_values.squeeze(axis=-1)
                shap_values = np.mean(np.abs(shap_values[:, :-2048]), axis=0)
                result[layer_name] = shap_values
        elif method == "sg":
            for layer_name in target_layers:
                print(f"Computing SHAP values for layer: {layer_name}")
                model = BINNExplainer(
                    self.binn, self.edge_indices, layer_name, self.device
                )
                baseline = model.get_tensor(background_data)
                inputs = model.get_tensor(test_data)
                explainer = GradientExplainer(model, baseline)
                shap_values = explainer.shap_values(inputs)
                shap_values = np.array(shap_values)
                if shap_values.ndim == 3:
                    shap_values = shap_values.squeeze(axis=-1)
                shap_values = np.mean(np.abs(shap_values[:, :-2048]), axis=0)
                result[layer_name] = shap_values
        else:
            raise ValueError(f"Invalid explain method: {method}")

        return result


class BINNExplainer(nn.Module):

    def __init__(
        self,
        model: BINNLayer,
        edge_indices: tuple[Tensor, ...],
        layer: str,
        device: torch.device,
    ):
        super().__init__()
        self.model = model.eval().to(device)
        self.gene_go, self.go_ke, self.ke_ke, self.tissue = move_device(
            *edge_indices, device=device
        )
        self.layer = layer

    def forward(self, input_tensor: Tensor) -> Tensor:
        if self.layer == "gene":
            return self.from_gene(input_tensor)
        elif self.layer == "go":
            return self.from_go(input_tensor)
        elif self.layer == "ke":
            return self.from_ke(input_tensor)
        raise ValueError("Invalid layer name")

    def get_tensor(self, input_tensor: Tensor) -> Tensor:
        if self.layer == "gene":
            return input_tensor
        elif self.layer == "go":
            _, go = self.model.gene2go(input_tensor[:, :-2048], self.gene_go)
            return torch.cat((go, input_tensor[:, -2048:]), dim=-1)
        elif self.layer == "ke":
            _, go = self.model.gene2go(input_tensor[:, :-2048], self.gene_go)
            _, ke = self.model.go2ke(go, self.go_ke)
            for layer in self.model.ke2ke:
                _, ke = layer(ke, self.ke_ke)
            return torch.cat((ke, input_tensor[:, -2048:]), dim=-1)
        raise ValueError("Invalid layer name")

    def public_forward(self, ke: Tensor, c: Tensor) -> Tensor:
        bio_result = self.model.bio(ke[:, self.tissue])
        drug_result = self.model.drug(c)
        result = torch.cat([bio_result, drug_result], dim=1)
        return self.model.predict(result)

    def from_gene(self, gene: Tensor) -> Tensor:
        y1, go = self.model.gene2go(gene[:, :-2048], self.gene_go)
        y2, ke = self.model.go2ke(go, self.go_ke)
        for layer in self.model.ke2ke:
            y3, ke = layer(ke, self.ke_ke)
        y4 = self.public_forward(ke, gene[:, -2048:])
        return (y1 + y2 + y3 + y4) / 4  # type: ignore

    def from_go(self, go: Tensor) -> Tensor:
        y2, ke = self.model.go2ke(go[:, :-2048], self.go_ke)
        for layer in self.model.ke2ke:
            y3, ke = layer(ke, self.ke_ke)
        y4 = self.public_forward(ke, go[:, -2048:])
        return (y2 + y3 + y4) / 3  # type: ignore

    def from_ke(self, ke: Tensor) -> Tensor:
        return self.public_forward(ke[:, :-2048], ke[:, -2048:])

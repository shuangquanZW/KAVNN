import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import Metric
from torchmetrics import AUROC, Accuracy, Precision, Recall, F1Score, HammingDistance
from torch_scatter import scatter

import pytorch_lightning as pl
from transformers import get_linear_schedule_with_warmup  # 用于学习率预热

from dl.VNN.VNNLayer import VNNLayer, VNNLoss
from dl.util_dl import (
    load_data,
    create_tensor,
    move_device,
)


def select_tensor(data: np.ndarray, index: list[int]) -> Tensor:
    return torch.from_numpy(data[index]).float()


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


class VNNDataModule(pl.LightningDataModule):

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


class VNN(pl.LightningModule):

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
        self.vnn = VNNLayer(
            num_gene_go, num_go, num_go_ke, num_ke, num_ke_ke, num_neurals, num_labels
        )
        # 记录训练步数，用于学习率预热
        self.total_steps = 0
        # 损失函数
        self.loss_function = VNNLoss(aggregation)
        # 评估器
        self.ham_evaluator = HammingDistance("binary")
        self.auc_evaluator = AUROC("binary")
        self.acc_evaluator = Accuracy("binary")
        self.precision_evaluator = Precision("binary")
        self.recall_evaluator = Recall("binary")
        self.f1_evaluator = F1Score("binary")
        # explain
        self.explain_model = None

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
    ) -> tuple[Tensor, Tensor]:
        return self.vnn(gene, gene_go, go_ke, ke_ke, tissue, compound)

    def training_step(self, batch: Tensor) -> Tensor:
        gene, compound_state, label = batch
        state, pred = self(gene, *self.edge_indices, compound_state)
        loss = self.loss_function(state, pred, label)
        self.log(
            "train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )
        # 记录训练步数，用于学习率预热
        self.total_steps += 1
        return loss

    def validation_step(self, batch: Tensor) -> None:
        gene, compound_state, label = batch
        state, pred = self(gene, *self.edge_indices, compound_state)
        loss = self.loss_function(state, pred, label)
        label = label.long()
        ham = self.ham_evaluator(pred, label)
        self.log("val_loss", loss, prog_bar=True, logger=True, on_epoch=True)
        self.log("val_ham", ham, prog_bar=True, logger=True, on_epoch=True)

    def test_step(self, batch: Tensor) -> dict[str, float]:
        gene, compound_state, label = batch
        _, pred = self(gene, *self.edge_indices, compound_state)
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
        _, pred = self(gene, *self.edge_indices, compound_state)
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

    def explain(self, drug: str) -> dict[str, dict[int, float]] | None:
        drug = drug.split("_")[0]
        expr, compound, _ = load_data()
        expr, compound = create_tensor(expr, compound, device=self.device)
        # get test data
        metadata = pd.read_csv("./data/source/metadata.csv")
        metadata = metadata[
            (metadata["dose"] != "control") & (metadata["compound"] == drug)
        ]
        index = metadata["index"].tolist()
        if not index:
            print(f"No test data found for drug: {drug}")
            return None
        expr = expr[index]
        compound = compound[index]
        self.edge_indices = move_device(*self.edge_indices, device=self.device)
        model = self.vnn.eval().to(self.device)
        state_pred, pred = model.get_states(
            expr, compound, *self.edge_indices
        )  # type: ignore
        pred = pred.detach().cpu().squeeze().numpy()
        state_pred = state_pred.detach().cpu().numpy()
        ridge = Ridge(alpha=0.3, fit_intercept=True)
        ridge.fit(state_pred, pred)
        coefs = torch.tensor(ridge.coef_, dtype=torch.float32, device=self.device)
        num_genes = expr.shape[1]
        num_go = self.vnn.gene2go.num_nodes
        num_ke = self.vnn.go2ke.num_nodes
        gene_coefs = coefs[:num_genes]
        go_coefs = coefs[num_genes : num_genes + num_go]
        ke_coefs = coefs[num_genes + num_go : num_genes + num_go + num_ke]
        gene_go, go_ke, ke_ke, _ = self.edge_indices
        go_childern = scatter(gene_coefs[gene_go[0]], gene_go[1], dim_size=num_go)
        go_childern[go_childern == 0] = 1
        ke_childern = scatter(go_coefs[go_ke[0]], go_ke[1], dim_size=num_ke)
        ke_childern[ke_childern == 0] = 1
        ke_ke_childern = scatter(ke_coefs[ke_ke[0]], ke_ke[1], dim_size=num_ke)
        ke_ke_childern[ke_ke_childern == 0] = 1
        go_coefs = go_coefs / go_childern
        ke_coefs = ke_coefs / ke_childern / ke_ke_childern
        gene_dict = {i: float(gene_coefs[i].item()) for i in range(len(gene_coefs))}
        go_dict = {i: float(go_coefs[i].item()) for i in range(len(go_coefs))}
        ke_dict = {i: float(ke_coefs[i].item()) for i in range(len(ke_coefs))}
        return {"gene": gene_dict, "go": go_dict, "ke": ke_dict}

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
        model = self.vnn.eval().to(self.device)
        self.edge_indices = move_device(*self.edge_indices, device=self.device)
        # get result
        baseline_result = model(baseline, *self.edge_indices, compound)[1]
        test_result = model(expr, *self.edge_indices, compound)[1]
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

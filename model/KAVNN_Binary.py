import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import AUROC, Accuracy, Precision, Recall, F1Score, HammingDistance

import pytorch_lightning as pl
from transformers import get_linear_schedule_with_warmup  # 用于学习率预热
from shap import DeepExplainer

from model.Loss import KAVNNLoss
from model.KAVNNLayers import KAVNNLayer
from model.Attribution import *
from util import move_device, load_data, create_tensor


def select_tensor(
    data: np.ndarray, index: list[int], dtype: torch.dtype = torch.float32
) -> Tensor:
    return torch.tensor(data[index], dtype=dtype)


def move_tensor(data: np.ndarray, dtype: torch.dtype = torch.float32):
    return torch.tensor(data, dtype=dtype)


class DataModule(pl.LightningDataModule):

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
            expr_pred = np.load(self.data_dir + "/drug_matrix.npy")
            comp_pred = np.load(self.data_dir + "/drug_matrix_compound.npy")
            label_pred = np.load(self.data_dir + "/drug_matrix_label.npy")
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
            self.predict_dataset = TensorDataset(
                move_tensor(expr_pred),
                move_tensor(comp_pred),
                move_tensor(label_pred),
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


class KA_VNN(pl.LightningModule):

    def __init__(
        self,
        num_go: int,
        num_ke: int,
        num_neurals: int = 2,
        grid_size: int = 2,
        bias: bool = True,
        aggregation: str = "mean",
        mode: str = "standard",
        num_labels: int = 1,
        alpha: float = 0.3,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.kan = KAVNNLayer(
            num_go, num_ke, num_neurals, grid_size, bias, aggregation, mode, num_labels
        )
        # 记录训练步数，用于学习率预热
        self.total_steps = 0
        # 损失函数
        self.loss_function = KAVNNLoss(aggregation, alpha)
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
    ) -> tuple[Tensor, Tensor]:
        return self.kan(gene, gene_go, go_ke, ke_ke, tissue, compound)

    def training_step(self, batch: Tensor) -> Tensor:
        gene, compound_state, label = batch
        pred, state_pred = self(gene, *self.edge_indices, compound_state)
        loss = self.loss_function(pred, state_pred, label)
        self.log(
            "train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )
        # 记录训练步数，用于学习率预热
        self.total_steps += 1
        return loss

    def validation_step(self, batch: Tensor) -> None:
        gene, compound_state, label = batch
        pred, state_pred = self(gene, *self.edge_indices, compound_state)
        loss = self.loss_function(pred, state_pred, label)
        label = label.long()
        ham = self.ham_evaluator(pred, label)
        self.log("val_loss", loss, prog_bar=True, logger=True, on_epoch=True)
        self.log("val_ham", ham, prog_bar=True, logger=True, on_epoch=True)

    def test_step(self, batch: Tensor) -> dict[str, float]:
        gene, compound_state, label = batch
        pred, _ = self(gene, *self.edge_indices, compound_state)
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
        pred, _ = self(gene, *self.edge_indices, compound_state)
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

    def get_states(self, expr: Tensor, compound: Tensor) -> Tensor:
        self.edge_indices = move_device(*self.edge_indices, device=self.device)
        self.kan.eval().to(self.device)
        with torch.no_grad():
            _, state_pred = self.kan.get_states(expr, *self.edge_indices, compound)  # type: ignore
        return state_pred

    def shap_explain(self, drug_batch: str) -> dict[str, np.ndarray] | None:
        drug, batch = drug_batch.split("_")
        background_data = self.get_background_data(batch)
        # preprocess data
        expr, _, _, compound = load_data()
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
        compound_batch = compound[index]
        test_data = torch.cat((test_data, compound_batch), dim=-1)
        compound_expanded = (
            compound_batch[0].unsqueeze(0).expand(background_data.shape[0], -1)
        )
        background_data = torch.cat((background_data, compound_expanded), dim=-1)
        target_layers = ["gene", "go", "ke"]
        models_dict = {
            "gene": KAVNN_Gene(self.kan, self.edge_indices, self.device),
            "go": KAVNN_GO(self.kan, self.edge_indices, self.device),
            "ke": KAVNN_KE(self.kan, self.edge_indices, self.device),
        }
        result = {}
        for layer_name in target_layers:
            print(f"Computing SHAP values for layer: {layer_name}")
            model = models_dict[layer_name]
            baseline = model.get_tensor(background_data)
            inputs = model.get_tensor(test_data)
            explainer = DeepExplainer(model, baseline)
            shap_values = explainer.shap_values(inputs, check_additivity=False)
            shap_values = np.array(shap_values).squeeze(axis=-1)
            shap_values = np.mean(np.abs(shap_values[:, :-2048]), axis=0)
            result[layer_name] = shap_values
        return result

    def ridge_explain(self, drug: str) -> dict[str, dict[int, float]] | None:
        drug = drug.split("_")[0]
        expr, _, _, compound = load_data()
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
        model = self.kan.eval().to(self.device)
        pred, state_pred = model.get_states(
            expr, *self.edge_indices, compound  # type: ignore
        )
        pred = pred.detach().cpu().squeeze().numpy()
        state_pred = state_pred.detach().cpu().numpy()
        ridge = Ridge(alpha=0.3, fit_intercept=True)
        ridge.fit(state_pred, pred)
        coefs = torch.tensor(ridge.coef_, dtype=torch.float32, device=self.device)
        num_genes = expr.shape[1]
        num_go = self.kan.gene2go.num_nodes
        num_ke = self.kan.go2ke.num_nodes
        gene_coefs = coefs[:num_genes]
        go_coefs = coefs[num_genes : num_genes + num_go]
        ke_coefs = coefs[num_genes + num_go : num_genes + num_go + num_ke]
        gene_go, go_ke, ke_ke, _ = self.edge_indices
        go_childern = scatter(gene_coefs[gene_go[0]], gene_go[1], dim_size=num_go)
        go_childern[go_childern == 0] = 1
        ke_childern = scatter(go_coefs[go_ke[0]], go_ke[1], dim_size=num_ke)
        ke_childern[ke_childern < 0.2] = 1
        ke_ke_childern = scatter(ke_coefs[ke_ke[0]], ke_ke[1], dim_size=num_ke)
        ke_ke_childern[ke_ke_childern < 0.2] = 1
        go_coefs = go_coefs / go_childern
        ke_coefs = ke_coefs / ke_childern / ke_ke_childern
        gene_dict = {i: float(gene_coefs[i].item()) for i in range(len(gene_coefs))}
        go_dict = {i: float(go_coefs[i].item()) for i in range(len(go_coefs))}
        ke_dict = {i: float(ke_coefs[i].item()) for i in range(len(ke_coefs))}
        return {"gene": gene_dict, "go": go_dict, "ke": ke_dict}

    def kan_hier_explain(self, drug_batch: str) -> dict[str, np.ndarray] | None:
        drug, batch = drug_batch.split("_")
        background_data = self.get_background_data(batch)
        self.edge_indices = move_device(*self.edge_indices, device=self.device)
        expr, _, _, compound = load_data()
        expr, compound = create_tensor(expr, compound, device=self.device)
        metadata = pd.read_csv("./data/source/metadata.csv")
        metadata = metadata[
            (metadata["dose"] != "control") & (metadata["compound"] == drug)
        ]
        index = metadata["index"].tolist()
        if not index:
            print(f"No test data found for drug & batch: {drug_batch}")
            return None
        test_data = expr[index]
        compound_batch = compound[index]
        test_full = torch.cat((test_data, compound_batch), dim=-1)
        compound_expanded = (
            compound_batch[0].unsqueeze(0).expand(background_data.shape[0], -1)
        )
        baseline_full = torch.cat((background_data, compound_expanded), dim=-1)
        baseline_full = torch.mean(baseline_full, dim=0).unsqueeze(0)

        from model.KANHierAttribution import cascade_gene_go_ke

        result = {}
        gene_s, go_s, ke_s = cascade_gene_go_ke(
            self, test_full, baseline_full, self.edge_indices
        )
        result["gene"] = gene_s.detach().cpu().numpy()
        result["go"] = go_s.detach().cpu().numpy()
        result["ke"] = ke_s.detach().cpu().numpy()

        return result

    def compute_attribution(
        self, drug_batch: str, surrogate: LogisticSurrogate
    ) -> dict[str, dict[int, float]] | None:
        drug, _ = drug_batch.split("_")
        # preprocess
        expr, label, _, compound = load_data()
        expr, compound = create_tensor(expr, compound, device=self.device)
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
        label = label[index]

        state_pred = self.get_states(expr, compound)
        state_pred = state_pred.detach().cpu().numpy()
        beta = surrogate.coef_
        contrib = (state_pred * beta).mean(axis=0)
        contrib = torch.from_numpy(contrib).to(self.device)

        num_genes = 15375
        num_go = self.kan.gene2go.num_nodes
        num_ke = self.kan.go2ke.num_nodes
        self.edge_indices = move_device(*self.edge_indices, device=self.device)
        gene_contrib = contrib[:num_genes]
        go_contrib = contrib[num_genes : num_genes + num_go]
        ke_contrib = contrib[num_genes + num_go : num_genes + num_go + num_ke]

        go_contrib = hierarchical_propagation(
            gene_contrib, go_contrib, self.edge_indices[0], num_go
        )
        ke_contrib = hierarchical_propagation(
            go_contrib, ke_contrib, self.edge_indices[1], num_ke
        )
        ke_contrib = hierarchical_propagation(
            ke_contrib, ke_contrib, self.edge_indices[2], num_ke
        )

        gene_contrib = normalization(gene_contrib)
        go_contrib = normalization(go_contrib)
        ke_contrib = normalization(ke_contrib)

        gene_dict = {i: float(gene_contrib[i].item()) for i in range(len(gene_contrib))}
        go_dict = {i: float(go_contrib[i].item()) for i in range(len(go_contrib))}
        ke_dict = {i: float(ke_contrib[i].item()) for i in range(len(ke_contrib))}

        return {"gene": gene_dict, "go": go_dict, "ke": ke_dict}

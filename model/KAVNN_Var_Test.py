import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import AUROC, Accuracy, Precision, Recall, F1Score, HammingDistance

import pytorch_lightning as pl
from transformers import get_linear_schedule_with_warmup  # 用于学习率预热

from model.Loss import KAVNNLoss
from model.KAVNN_Var import KAVNNLayer


def select_tensor(
    data: np.ndarray, index: list[int], dtype: torch.dtype = torch.float32
) -> Tensor:
    return torch.tensor(data[index], dtype=dtype)


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
        # state = np.load(self.data_dir + "/state.npy")[index]
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


class KA_VNN(pl.LightningModule):

    def __init__(
        self,
        num_go: int,
        num_ke: int,
        num_neurals: list[int] = [2, 2, 2],
        grid_sizes: list[int] = [2, 2, 2],
        reshape_sizes: list[int] = [32, 2],
        bias: bool = True,
        aggregation: str = "mean",
        num_labels: int = 1,
        alpha: float = 0.3,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.kan = KAVNNLayer(
            num_go,
            num_ke,
            num_neurals,
            grid_sizes,
            reshape_sizes,
            bias,
            aggregation,
            num_labels,
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

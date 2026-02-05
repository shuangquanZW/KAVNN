import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch import Tensor, nn
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import Metric
from torchmetrics import AUROC, Accuracy, Precision, Recall, F1Score, HammingDistance

import pytorch_lightning as pl
from transformers import get_linear_schedule_with_warmup  # 用于学习率预热

from dl.DNN.DNNLayer import DNNLayer


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


class DNNDataModule(pl.LightningDataModule):

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


class DNN(pl.LightningModule):

    def __init__(
        self,
        num_neurals: int = 8,
        aggregation: str = "mean",
        lr: float = 1e-4,
        num_labels: int = 1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.dnn = DNNLayer(num_neurals, num_labels)
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
        pass

    def forward(
        self,
        gene: Tensor,
        compound: Tensor,
    ) -> Tensor:
        return self.dnn(gene, compound)

    def training_step(self, batch: Tensor) -> Tensor:
        gene, compound_state, label = batch
        pred = self(gene, compound_state)
        loss = self.loss_function(pred, label)
        self.log(
            "train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )
        # 记录训练步数，用于学习率预热
        self.total_steps += 1
        return loss

    def validation_step(self, batch: Tensor) -> None:
        gene, compound_state, label = batch
        pred = self(gene, compound_state)
        loss = self.loss_function(pred, label)
        label = label.long()
        ham = self.ham_evaluator(pred, label)
        self.log("val_loss", loss, prog_bar=True, logger=True, on_epoch=True)
        self.log("val_ham", ham, prog_bar=True, logger=True, on_epoch=True)

    def test_step(self, batch: Tensor) -> dict[str, float]:
        gene, compound_state, label = batch
        pred = self(gene, compound_state)
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
        pred = self(gene, compound_state)
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

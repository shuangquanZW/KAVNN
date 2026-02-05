from typing import Tuple, Dict, Callable, Any, List
import time
import pickle as pkl

import torch
from torch import Tensor, nn
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)


EXPR = "./data/expr.npy"
COMP = "./data/compound.npy"
STA = "./data/state.npy"
LABEL = "./data/label.npy"
LIVER = "./data/edge_index/liver_index.npy"
KIDNEY = "./data/edge_index/kidney_index.npy"
GENE_GO = "./data/edge_index/gene_go_bp.npy"
GO_KE = "./data/edge_index/go_ke_bp.npy"
KE_KE = "./data/edge_index/ke_ke.npy"
TISSUE = "./data/edge_index/tissue.npy"
BASELINE_INDEX = "./data/pkl/baseline_index.pkl"


def load_data(organ: str | None = None) -> Tuple[np.ndarray[Any, Any], ...]:
    if organ == "liver":
        index = np.load(LIVER)
    elif organ == "kidney":
        index = np.load(KIDNEY)
    else:
        index = np.arange(5750)
    expr = np.load(EXPR)[index]
    comp = np.load(COMP)[index]
    label = np.load(LABEL)[index]
    return expr, comp, label


def select_tensor(data: np.ndarray, index: list[int]) -> Tensor:
    return torch.tensor(data[index], dtype=torch.float32)


def create_tensor(*data: np.ndarray[Any, Any], device: torch.device) -> List[Tensor]:
    return [
        torch.tensor(d, dtype=torch.float32, requires_grad=True, device=device)
        for d in data
    ]


def move_device(*args: Tensor, device: torch.device) -> Tuple[Tensor, ...]:
    return tuple(x.to(device) for x in args)


def get_edge_indices() -> Tuple[Tensor, ...]:
    # 加载边索引文件
    gene_go = np.load(GENE_GO)
    go_ke = np.load(GO_KE)
    ke_ke = np.load(KE_KE)
    tissue = np.load(TISSUE)
    # 转化为tensor
    gene_go = torch.tensor(gene_go, dtype=torch.long)
    go_ke = torch.tensor(go_ke, dtype=torch.long)
    ke_ke = torch.tensor(ke_ke, dtype=torch.long)
    tissue = torch.tensor(tissue, dtype=torch.long)
    return gene_go, go_ke, ke_ke, tissue


def split_data_index(
    random_state: int, organ: str | None = None
) -> Tuple[list[int], list[int], list[int]]:
    _, _, label = load_data(organ)
    length = label.shape[0]
    index = np.arange(length)
    train, no_train = train_test_split(
        index,
        test_size=0.4,
        random_state=random_state,
    )
    valid, test = train_test_split(
        no_train,
        test_size=0.5,
        random_state=random_state,
    )
    return list(train), list(valid), list(test)


def preprocess_data(
    random_state: int,
    batch_size_train: int,
    batch_size_eval: int,
    organ: str | None = None,
) -> Tuple[
    DataLoader[Tuple[Tensor, ...]],
    DataLoader[Tuple[Tensor, ...]],
    DataLoader[Tuple[Tensor, ...]],
]:
    expr, comp_sta, label = load_data(organ)
    train_index, valid_index, test_index = split_data_index(random_state, organ)

    train_dataset = TensorDataset(
        select_tensor(expr, train_index),
        select_tensor(comp_sta, train_index),
        select_tensor(label, train_index),
    )
    val_dataset = TensorDataset(
        select_tensor(expr, valid_index),
        select_tensor(comp_sta, valid_index),
        select_tensor(label, valid_index),
    )
    test_dataset = TensorDataset(
        select_tensor(expr, test_index),
        select_tensor(comp_sta, test_index),
        select_tensor(label, test_index),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_eval, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_eval, shuffle=False)

    return train_loader, val_loader, test_loader


def evaluate(y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any]) -> None:
    auc = float(np.round(roc_auc_score(y_true, y_pred), 4))
    y_pred = np.where(y_pred >= 0.5, 1, 0)
    acc = float(np.round(accuracy_score(y_true, y_pred), 4))
    pre = float(np.round(precision_score(y_true, y_pred), 4))
    rec = float(np.round(recall_score(y_true, y_pred), 4))
    f1 = float(np.round(f1_score(y_true, y_pred), 4))
    print(f"AUC: {auc}, ACC: {acc}, PR: {pre}, RE: {rec}, F1: {f1}")


def get_metrics(
    y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any]
) -> Tuple[float, float, float, float, float]:
    auc = float(np.round(roc_auc_score(y_true, y_pred), 4))
    y_pred = (y_pred >= 0.5).astype(int)
    acc = float(np.round(accuracy_score(y_true, y_pred), 4))
    pre = float(np.round(precision_score(y_true, y_pred), 4))
    rec = float(np.round(recall_score(y_true, y_pred), 4))
    f1 = float(np.round(f1_score(y_true, y_pred), 4))
    return auc, acc, pre, rec, f1


def calculate_metrics_stats(data: list[list[float]]) -> Dict[str, Tuple[float, float]]:
    metrics_array = np.array(data, dtype=float)
    metrics_names = ["auc", "acc", "pre", "rec", "f1"]
    return {
        name: (
            round(metrics_array[:, i].mean(), 4),
            round(metrics_array[:, i].std(), 4),
        )
        for i, name in enumerate(metrics_names)
    }


def make_mlp(neural_count: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LazyLinear(512),
        nn.Tanh(),
        nn.BatchNorm1d(512),
        nn.Linear(512, 256),
        nn.Tanh(),
        nn.BatchNorm1d(256),
        nn.Linear(256, 64),
        nn.Tanh(),
        nn.BatchNorm1d(64),
        nn.Linear(64, neural_count),
    )

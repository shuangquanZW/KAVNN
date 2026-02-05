from typing import Any
import numpy as np
import pickle as pkl

import torch
from torch import Tensor


def move_device(*args: Tensor, device: torch.device) -> tuple[Tensor, ...]:
    return tuple(x.to(device) for x in args)


def load_data() -> tuple[np.ndarray[Any, Any], ...]:
    expr = np.load("./data/expr.npy")
    label = np.load("./data/label.npy")
    tissue = np.load("./data/edge_index/tissue.npy")
    compound = np.load("./data/compound.npy")
    return expr, label, tissue, compound


def create_tensor(
    *data: np.ndarray[Any, Any], device: str | torch.device
) -> list[Tensor]:
    return [
        torch.tensor(d, dtype=torch.float32, requires_grad=True, device=device)
        for d in data
    ]


def load_baseline_dict() -> dict[tuple[str, str, str, str], list[int]]:
    with open("./data/pkl/baseline_index.pkl", "rb") as f:
        return pkl.load(f)


def get_baseline_index(
    baseline_dict: dict[tuple[str, str, str, str], list[int]], comp_name: str
) -> list[int]:
    for (name, _, _, _), indices in baseline_dict.items():
        if name == comp_name:
            return indices
    return []


def get_sample_index(
    baseline_dict: dict[tuple[str, str, str, str], list[int]], comp_name: str
) -> list[int]:
    return [
        idx
        for (name, level, _, _), indices in baseline_dict.items()
        if name == comp_name and level != "control"
        for idx in indices
    ]


def get_edge_indices() -> tuple[Tensor, ...]:
    # 加载边索引文件
    gene_go = np.load("./data/edge_index/gene_go_bp.npy")
    go_ke = np.load("./data/edge_index/ke_ke.npy")
    ke_ke = np.load("./data/edge_index/ke_ke.npy")
    tissue = np.load("./data/edge_index/tissue.npy")
    # 转化为tensor
    gene_go = torch.tensor(gene_go, dtype=torch.long)
    go_ke = torch.tensor(go_ke, dtype=torch.long)
    ke_ke = torch.tensor(ke_ke, dtype=torch.long)
    tissue = torch.tensor(tissue, dtype=torch.long)
    return gene_go, go_ke, ke_ke, tissue

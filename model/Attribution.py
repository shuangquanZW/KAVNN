import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import torch
from torch import Tensor, nn
from torch_scatter import scatter
from model.KAVNNLayers import KAVNNLayer
from util import move_device


class KAVNN_Gene(nn.Module):

    def __init__(
        self,
        model: KAVNNLayer,
        edge_indices: tuple[Tensor, ...],
        device: torch.device,
    ) -> None:
        super().__init__()
        self.kavnn = model.eval().to(device)
        self.gene_go, self.go_ke, self.ke_ke, self.tissue = move_device(
            *edge_indices, device=device
        )

    def get_tensor(self, input_tensor: Tensor) -> Tensor:
        return input_tensor

    def forward(self, input_tensor: Tensor) -> Tensor:
        gene = self.kavnn.gene_layer(input_tensor[:, :-2048].unsqueeze(-1))
        go = self.kavnn.gene2go(gene, self.gene_go)
        go = self.kavnn.go_encode_layer(go)
        go = self.kavnn.go_decode_layer(go.unsqueeze(-1))
        ke = self.kavnn.go2ke(go, self.go_ke)
        for layer in self.kavnn.ke2ke:
            ke = layer(ke, self.ke_ke)
        ke = self.kavnn.ke_layer(ke)
        bio_pred = ke[:, self.tissue]
        for layer in self.kavnn.bio:
            bio_pred = layer(bio_pred)
        drug_pred = input_tensor[:, -2048:]
        for layer in self.kavnn.drug:
            drug_pred = layer(drug_pred)
        combined_features = torch.cat([bio_pred, drug_pred], dim=-1)
        pred = self.kavnn.predictor(combined_features)
        return pred


class KAVNN_GO(nn.Module):

    def __init__(
        self,
        model: KAVNNLayer,
        edge_indices: tuple[Tensor, ...],
        device: torch.device,
    ) -> None:
        super().__init__()
        self.kavnn = model.eval().to(device)
        self.gene_go, self.go_ke, self.ke_ke, self.tissue = move_device(
            *edge_indices, device=device
        )

    def get_tensor(self, input_tensor: Tensor) -> Tensor:
        gene = self.kavnn.gene_layer(input_tensor[:, :-2048].unsqueeze(-1))
        go = self.kavnn.gene2go(gene, self.gene_go)
        go = self.kavnn.go_encode_layer(go)
        return torch.cat((go, input_tensor[:, -2048:]), dim=-1)

    def forward(self, input_tensor: Tensor) -> Tensor:
        go = self.kavnn.go_decode_layer(input_tensor[:, :-2048].unsqueeze(-1))
        ke = self.kavnn.go2ke(go, self.go_ke)
        for layer in self.kavnn.ke2ke:
            ke = layer(ke, self.ke_ke)
        ke = self.kavnn.ke_layer(ke)
        bio_pred = ke[:, self.tissue]
        for layer in self.kavnn.bio:
            bio_pred = layer(bio_pred)
        drug_pred = input_tensor[:, -2048:]
        for layer in self.kavnn.drug:
            drug_pred = layer(drug_pred)
        combined_features = torch.cat([bio_pred, drug_pred], dim=-1)
        pred = self.kavnn.predictor(combined_features)
        return pred


class KAVNN_KE(nn.Module):

    def __init__(
        self,
        model: KAVNNLayer,
        edge_indices: tuple[Tensor, ...],
        device: torch.device,
    ) -> None:
        super().__init__()
        self.kavnn = model.eval().to(device)
        self.gene_go, self.go_ke, self.ke_ke, self.tissue = move_device(
            *edge_indices, device=device
        )

    def get_tensor(self, input_tensor: Tensor) -> Tensor:
        gene = self.kavnn.gene_layer(input_tensor[:, :-2048].unsqueeze(-1))
        go = self.kavnn.gene2go(gene, self.gene_go)
        go = self.kavnn.go_encode_layer(go)
        go = self.kavnn.go_decode_layer(go.unsqueeze(-1))
        ke = self.kavnn.go2ke(go, self.go_ke)
        ke = self.kavnn.go2ke(go, self.go_ke)
        for layer in self.kavnn.ke2ke:
            ke = layer(ke, self.ke_ke)
        ke = self.kavnn.ke_layer(ke)
        return torch.cat((ke, input_tensor[:, -2048:]), dim=-1)

    def forward(self, input_tensor: Tensor) -> Tensor:
        bio_pred = input_tensor[:, :-2048][:, self.tissue]
        for layer in self.kavnn.bio:
            bio_pred = layer(bio_pred)
        drug_pred = input_tensor[:, -2048:]
        for layer in self.kavnn.drug:
            drug_pred = layer(drug_pred)
        combined_features = torch.cat([bio_pred, drug_pred], dim=-1)
        pred = self.kavnn.predictor(combined_features)
        return pred


class LogisticSurrogate:

    def __init__(self):
        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l2",
                        solver="saga",
                        max_iter=2000,
                        class_weight="balanced",
                        C=0.5,
                    ),
                ),
            ]
        )

    def fit(self, states: np.ndarray, labels: np.ndarray):
        self.model.fit(states, labels)

    @property
    def coef_(self) -> np.ndarray:
        clf = self.model.named_steps["clf"]
        scaler = self.model.named_steps["scaler"]
        return clf.coef_.squeeze() / scaler.scale_

    @property
    def intercept_(self) -> float:
        return self.model.named_steps["clf"].intercept_[0]


def hierarchical_propagation(
    child_contrib: Tensor,
    parent_contrib: Tensor,
    edge_index: Tensor,
    num_parents: int,
    alpha: float = 0.5,
) -> Tensor:
    src, dst = edge_index
    child_contrib = scatter(
        child_contrib[src], dst, dim_size=num_parents, reduce="mean"
    )
    parent_contrib = alpha * parent_contrib + (1 - alpha) * child_contrib
    return parent_contrib


def normalization(contrib: Tensor) -> Tensor:
    std = torch.std(contrib) + 1e-8
    return torch.tanh(contrib / std)

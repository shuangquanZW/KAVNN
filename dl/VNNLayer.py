import torch
from torch import nn, Tensor
from torch_scatter import scatter


def make_mlp(neural_count: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LazyLinear(512),
        nn.Sigmoid(),
        nn.BatchNorm1d(512),
        nn.Linear(512, 256),
        nn.Sigmoid(),
        nn.BatchNorm1d(256),
        nn.Linear(256, 64),
        nn.Sigmoid(),
        nn.BatchNorm1d(64),
        nn.Linear(64, neural_count),
    )


class StateActivate(nn.Module):

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.get_state = nn.Linear(input_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.get_state(x).squeeze(-1)


class VnnLayer(nn.Module):

    def __init__(self, num_edges: int, num_nodes: int, num_neurals: int) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.weight = nn.Parameter(torch.randn(num_edges, num_neurals, num_neurals))
        self.bias = nn.Parameter(torch.randn(num_edges, num_neurals))
        self.dropout = nn.Dropout(0.5)
        self.bn = nn.LazyBatchNorm1d()
        self.state = StateActivate(num_neurals)

    def forward(self, x: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        edge_index = edge_index.to(x.device)
        src, dst = edge_index
        x = self.dropout(x[:, src, :])
        x = torch.einsum("bei,eio->beo", x, self.weight) + self.bias.unsqueeze(0)
        x = torch.tanh(x)
        x = scatter(x, dst, dim=1, reduce="mean", dim_size=self.num_nodes)
        state = self.state(x)
        return state, self.bn(x)


class VNNLayer(nn.Module):

    def __init__(
        self,
        num_gene_go: int,
        num_go: int,
        num_go_ke: int,
        num_ke: int,
        num_ke_ke: int,
        num_neural: int = 8,
        num_labels: int = 20,
    ) -> None:
        super().__init__()
        self.gene_layer = nn.Sequential(
            nn.Linear(1, num_neural),
            nn.Tanh(),
            nn.LazyBatchNorm1d(),
        )
        self.gene_state = StateActivate(num_neural)
        self.gene2go = VnnLayer(num_gene_go, num_go, num_neural)
        self.go2ke = VnnLayer(num_go_ke, num_ke, num_neural)
        self.ke2ke = nn.ModuleList(
            [VnnLayer(num_ke_ke, num_ke, num_neural) for _ in range(3)]
        )
        self.ke_layer = nn.Linear(num_neural, 1)
        self.bio = make_mlp(num_neural)
        self.drug = make_mlp(num_neural)
        self.reshaper = nn.LazyLinear(num_labels)
        self.predict = nn.Linear(num_neural * 2, num_labels)

    def forward(
        self,
        gene: Tensor,
        edge_gene_go: Tensor,
        edge_go_ke: Tensor,
        edge_ke_ke: Tensor,
        tissue: Tensor,
        c: Tensor,
    ) -> tuple[Tensor, Tensor]:
        assert gene.dim() == 2, "Input tensor must be (batch_size, num_nodes)"
        tissue = tissue.to(gene.device)
        gene = self.gene_layer(gene.unsqueeze(-1))
        gene_state = self.gene_state(gene)
        go_state, go = self.gene2go(gene, edge_gene_go)
        _, ke = self.go2ke(go, edge_go_ke)
        for layer in self.ke2ke:
            ke_state, ke = layer(ke, edge_ke_ke)
        bio_node_state = torch.cat([gene_state, go_state, ke_state], dim=-1)  # type: ignore
        ke = self.ke_layer(ke).squeeze(-1)
        bio_pred = self.bio(ke[:, tissue])
        drug_pred = self.drug(c)
        result = self.predict(torch.cat([bio_pred, drug_pred], dim=-1))
        return self.reshaper(bio_node_state), result

    def get_states(
        self,
        gene: Tensor,
        c: Tensor,
        edge_gene_go: Tensor,
        edge_go_ke: Tensor,
        edge_ke_ke: Tensor,
        tissue: Tensor,
    ) -> tuple[Tensor, Tensor]:
        assert gene.dim() == 2, "Input tensor must be (batch_size, num_nodes)"
        tissue = tissue.to(gene.device)
        gene = self.gene_layer(gene.unsqueeze(-1))
        gene_state = self.gene_state(gene)
        go_state, go = self.gene2go(gene, edge_gene_go)
        _, ke = self.go2ke(go, edge_go_ke)
        for layer in self.ke2ke:
            ke_state, ke = layer(ke, edge_ke_ke)
        bio_node_state = torch.cat([gene_state, go_state, ke_state], dim=-1)  # type: ignore
        ke = self.ke_layer(ke).squeeze(-1)
        bio_pred = self.bio(ke[:, tissue])
        drug_pred = self.drug(c)
        result = self.predict(torch.cat([bio_pred, drug_pred], dim=-1))
        return bio_node_state, result


class VNNLoss(nn.Module):

    def __init__(self, reduction: str = "mean", alpha: float = 0.3) -> None:
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, x: Tensor, y_hat: Tensor, y: Tensor) -> Tensor:
        root_loss = torch.binary_cross_entropy_with_logits(y_hat, y)
        node_loss = torch.binary_cross_entropy_with_logits(x, y)
        loss = root_loss + self.alpha * node_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

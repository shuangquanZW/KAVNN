import torch
from torch import nn, Tensor
from torch_scatter import scatter


def linear_func(x: Tensor, a: Tensor, b: Tensor) -> Tensor:
    return a * x + b


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


class BioLayer(nn.Module):

    def __init__(self, num_edges: int, num_nodes: int, num_labels: int = 20) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.randn(num_edges))
        self.bias = nn.Parameter(torch.randn(num_edges))
        self.predict = nn.LazyLinear(num_labels)
        self.bn = nn.LazyBatchNorm1d()
        self.num_nodes = num_nodes

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        src, dst = edge_index
        x_hat = linear_func(x[:, src], self.alpha, self.bias)
        x_hat = scatter(
            x_hat,
            dst,
            dim=1,
            reduce="mean",
            dim_size=self.num_nodes,
        )
        out = torch.tanh(x_hat)
        return self.predict(x_hat), self.bn(out)


class BINNLayer(nn.Module):

    def __init__(
        self,
        num_gene_go: int,
        num_go: int,
        num_go_ke: int,
        num_ke: int,
        num_ke_ke: int,
        neural_count: int = 8,
        num_labels: int = 20,
    ) -> None:
        super().__init__()
        self.gene2go = BioLayer(num_gene_go, num_go, num_labels)
        self.go2ke = BioLayer(num_go_ke, num_ke, num_labels)
        self.ke2ke = nn.ModuleList(
            [BioLayer(num_ke_ke, num_ke, num_labels) for _ in range(3)]
        )
        self.bio = make_mlp(neural_count)
        self.drug = make_mlp(neural_count)
        self.predict = nn.Linear(neural_count * 2, num_labels)

    def forward(
        self,
        gene: Tensor,
        edge_gene_go: Tensor,
        edge_go_ke: Tensor,
        edge_ke_ke: Tensor,
        tissue: Tensor,
        c: Tensor,
    ) -> Tensor:
        assert gene.dim() == 2, "Input tensor must be (batch_size, num_nodes)"
        y1, go = self.gene2go(gene, edge_gene_go)
        y2, ke = self.go2ke(go, edge_go_ke)
        for layer in self.ke2ke:
            y3, ke = layer(ke, edge_ke_ke)
        bio_result = self.bio(ke[:, tissue])
        drug_result = self.drug(c)
        result = torch.cat([bio_result, drug_result], dim=1)
        y4 = self.predict(result)
        return (y1 + y2 + y3 + y4) / 4  # type: ignore

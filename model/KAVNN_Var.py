import torch
from torch import nn, Tensor

from model.FourierKAN import FourierKANLayer, GraphFourierKANLayer


class NodeStateActivateLayer(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.get_node_state = nn.LazyLinear(1)

    def forward(self, x: Tensor) -> Tensor:
        return self.get_node_state(x).squeeze(-1)


class KAVNNLayer(nn.Module):

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
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        # node
        self.gene_layer = nn.Sequential(
            nn.Linear(1, num_neurals[0]),
            nn.Tanh(),
            nn.LazyBatchNorm1d(),
        )
        self.gene_state = NodeStateActivateLayer()
        self.go_encode_layer = NodeStateActivateLayer()
        self.go_decode_layer = nn.Sequential(
            nn.Linear(1, num_neurals[1]),
            nn.Tanh(),
        )
        self.go_state = NodeStateActivateLayer()
        self.ke_layer = NodeStateActivateLayer()
        self.ke_state = NodeStateActivateLayer()
        # aop
        self.gene2go = GraphFourierKANLayer(
            num_neurals[0], num_neurals[1], num_go, grid_sizes[0], bias, aggregation
        )
        self.go2ke = GraphFourierKANLayer(
            num_neurals[1], num_neurals[2], num_ke, grid_sizes[1], bias, aggregation
        )
        self.ke2ke = nn.ModuleList()
        for _ in range(2):
            self.ke2ke.append(
                GraphFourierKANLayer(
                    num_neurals[2],
                    num_neurals[2],
                    num_ke,
                    grid_sizes[2],
                    bias,
                    aggregation,
                    True,
                )
            )
        # concat
        readout_size = reshape_sizes[0]
        readout_grid = reshape_sizes[1]
        self.bio = nn.ModuleList()
        self.bio.append(nn.LazyLinear(readout_size))
        self.bio.append(
            FourierKANLayer(readout_size, readout_size // 2, readout_grid, bias)
        )
        self.drug = nn.ModuleList()
        self.drug.append(nn.LazyLinear(readout_size))
        self.drug.append(
            FourierKANLayer(readout_size, readout_size // 2, readout_grid, bias)
        )
        # predictor
        self.state_predictor = nn.LazyLinear(num_labels)
        self.predictor = FourierKANLayer(readout_size, num_labels, readout_grid, bias)

    def forward(
        self,
        gene: Tensor,
        gene_go: Tensor,
        go_ke: Tensor,
        ke_ke: Tensor,
        tissue: Tensor,
        compound: Tensor,
    ) -> tuple[Tensor, Tensor]:
        assert gene.dim() == 2, "The dim of Tensor gene is not 2!"
        # gene
        gene = self.gene_layer(gene.unsqueeze(-1))
        gene_state = self.gene_state(gene)
        # aop
        go = self.gene2go(gene, gene_go)
        go = self.go_encode_layer(go)
        go = self.go_decode_layer(go.unsqueeze(-1))
        go_state = self.go_state(go)
        ke = self.go2ke(go, go_ke)
        for layer in self.ke2ke:
            ke = layer(ke, ke_ke)
        ke_state = self.ke_state(ke)
        ke = self.ke_layer(ke)
        # node state
        state_pred = torch.cat([gene_state, go_state, ke_state], dim=-1)  # type: ignore
        state_pred = self.state_predictor(state_pred)
        # concat
        bio_pred = ke[:, tissue]
        for layer in self.bio:
            bio_pred = layer(bio_pred)
        drug_pred = compound
        for layer in self.drug:
            drug_pred = layer(drug_pred)
        combined_features = torch.cat([bio_pred, drug_pred], dim=-1)
        pred = self.predictor(combined_features)
        return pred, state_pred

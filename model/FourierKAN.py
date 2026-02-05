import math
import torch
from torch import nn, Tensor
from torch_scatter import scatter


class FourierKANLayer(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_size: int,
        bias: bool = True,
    ):
        super().__init__()
        # 傅里叶系数初始化
        self.fourier_coeffs = nn.Parameter(
            torch.randn(2, input_dim, grid_size, output_dim)
            / math.sqrt(grid_size * output_dim)  # 缩放初始化，避免梯度爆炸
        )
        self.bias = nn.Parameter(torch.zeros(1, output_dim)) if bias else None

        # 傅里叶频率项
        self.register_buffer(
            "fourier_freqs",
            torch.arange(1, grid_size + 1, dtype=torch.float32).view(1, 1, grid_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        # 计算傅里叶项
        kx = x.unsqueeze(-1) * self.fourier_freqs  # type: ignore
        cos_terms = torch.cos(kx)
        sin_terms = torch.sin(kx)

        # 傅里叶激活与求和
        cos_activated = cos_terms.unsqueeze(-1) * self.fourier_coeffs[0]
        sin_activated = sin_terms.unsqueeze(-1) * self.fourier_coeffs[1]

        cos_sum = cos_activated.sum(dim=-2)
        sin_sum = sin_activated.sum(dim=-2)

        y = (cos_sum + sin_sum).sum(dim=-2)

        if self.bias is not None:
            y = y + self.bias
        return y


class GraphFourierKANLayer(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_nodes: int,
        grid_size: int,
        bias: bool = True,
        aggregation: str = "mean",
        mode: str = "standard",
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.aggregation = aggregation
        self.mode = mode

        # 傅里叶系数初始化
        self.fourier_coeffs = nn.Parameter(
            torch.randn(2, num_nodes, input_dim, grid_size, output_dim)
            / math.sqrt(grid_size * output_dim)  # 缩放初始化，避免梯度爆炸
        )
        self.bias = (
            nn.Parameter(torch.zeros(1, num_nodes, output_dim)) if bias else None
        )

        # 傅里叶频率项
        self.register_buffer(
            "fourier_freqs",
            torch.arange(1, grid_size + 1, dtype=torch.float32).view(
                1, 1, 1, grid_size
            ),
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"输入形状必须为 [batch_size, num_nodes, input_dim]，当前为 {x.shape}"
            )
        _, num_nodes, _ = x.shape
        if self.mode == "standard":
            # 获取源节点和目标节点
            src, dst = edge_index
            # 邻居特征聚合
            x_hat = scatter(
                x[:, src, :],
                dst,
                dim=1,
                reduce=self.aggregation,
                dim_size=self.num_nodes,
            )  # [batch_size, num_nodes, output_dim]
        elif self.mode == "random":
            num_edges = edge_index.size(1)
            src = torch.randint(0, num_nodes, (num_edges,), device=x.device)
            dst = torch.randint(0, self.num_nodes, (num_edges,), device=x.device)
            x_hat = scatter(
                x[:, src, :],
                dst,
                dim=1,
                reduce=self.aggregation,
                dim_size=self.num_nodes,
            )
        elif self.mode == "full":
            if self.aggregation == "mean":
                # [batch, 1, input_dim] -> [batch, num_nodes, input_dim]
                x_hat = x.mean(dim=1, keepdim=True).expand(-1, self.num_nodes, -1)
            elif self.aggregation == "sum":
                x_hat = x.sum(dim=1, keepdim=True).expand(-1, self.num_nodes, -1)
        # KAN层
        kx = x_hat.unsqueeze(-1) * self.fourier_freqs  # type: ignore
        cos_terms = torch.cos(kx)
        sin_terms = torch.sin(kx)
        cos_activated = cos_terms.unsqueeze(-1) * self.fourier_coeffs[0]
        sin_activated = sin_terms.unsqueeze(-1) * self.fourier_coeffs[1]
        cos_sum = cos_activated.sum(dim=-2)
        sin_sum = sin_activated.sum(dim=-2)
        y = (cos_sum + sin_sum).sum(dim=-2)
        if self.bias is not None:
            y = y + self.bias
        return y

import torch
import torch.nn as nn
from torch import Tensor


class DNNLayer(nn.Module):

    def __init__(self, neural: int, num_labels: int = 20):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LazyLinear(1024),
            nn.Sigmoid(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.Sigmoid(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.Sigmoid(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.Sigmoid(),
            nn.BatchNorm1d(64),
            nn.Linear(64, neural),
        )
        self.predict = nn.Linear(neural, num_labels)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        x = self.mlp(torch.cat([x, c], dim=1))
        x_hat = self.predict(x)
        return x_hat

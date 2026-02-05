import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Attention3DBlock(nn.Module):
    def __init__(self, time_steps: int):
        super().__init__()
        self.attention_dense = nn.Linear(time_steps, time_steps)

    def forward(self, x):
        a = x.permute(0, 2, 1)
        a = F.relu(self.attention_dense(a))
        a = F.softmax(a, dim=-1)
        a_probs = a.permute(0, 2, 1)
        output = x * a_probs
        return output


class AttRethinkNet(nn.Module):
    def __init__(
        self,
        n_gene_features: int,
        n_fingerprint_features: int,
        n_labels: int,
        time_steps: int = 5,
        rnn_units: int = 128,
        dense_units: int = 128,
    ):
        super().__init__()
        self.time_steps = time_steps
        self.n_labels = n_labels
        total_input_dim = n_gene_features + n_fingerprint_features
        self.feature_projection = nn.Linear(total_input_dim, dense_units)
        self.attention = Attention3DBlock(time_steps)
        self.mid_dense = nn.Linear(dense_units, dense_units)
        self.lstm = nn.LSTM(
            input_size=dense_units, hidden_size=rnn_units, batch_first=True
        )
        self.classifier = nn.Linear(rnn_units, n_labels)

    def forward(self, gene_input: Tensor, fingerprint_input: Tensor) -> Tensor:
        x = torch.cat((gene_input, fingerprint_input), dim=-1)
        x = F.relu(self.feature_projection(x))
        x = x.unsqueeze(1).repeat(1, self.time_steps, 1)
        x = self.attention(x)
        x = F.relu(self.mid_dense(x))
        lstm_out, (_, _) = self.lstm(x)
        outputs = self.classifier(lstm_out)
        return outputs

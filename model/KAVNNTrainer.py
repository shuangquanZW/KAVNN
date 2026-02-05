# import argparse
# import numpy as np
# from sklearn.metrics import (
#     roc_auc_score,
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     hamming_loss,
# )

# import torch
# from torch import Tensor
# from torch.utils.data import DataLoader, TensorDataset

# from model.KAVNNLayers import KAVNNLayer
# from model.Loss import KAVNNLoss


# def select_tensor(
#     data: np.ndarray, index: list[int], dtype: torch.dtype = torch.float32
# ) -> Tensor:
#     return torch.tensor(data[index], dtype=dtype)


# class KAVNNTrainer:

#     def __init__(
#         self,
#         num_go: int,
#         num_ke: int,
#         num_neurals: int = 8,
#         grid_size: int = 2,
#         bias: bool = True,
#         aggregation: str = "mean",
#         num_labels: int = 1,
#         alpha: float = 0.3,
#         lr: float = 1e-3,
#         device: str = "cuda:0",
#     ):
#         self.device = device
#         self.kan = KAVNNLayer(
#             num_go, num_ke, num_neurals, grid_size, bias, aggregation, num_labels
#         )
#         self.loss_function = KAVNNLoss(aggregation, alpha)
#         self.optimizer = torch.optim.AdamW(self.kan.parameters(), lr=lr)

#     def init_dataset(self):
#         # load
#         expr = np.load("data/expr.npy")
#         compound = np.load("data/compound.npy")
#         state = np.load("data/state.npy")
#         label = np.load("data/label.npy")
#         compound_state = np.concatenate([compound, state], axis=1)
#         # index
#         index = [i for i in range(len(label))]
#         # train_dataset
#         train_dataset = TensorDataset(
#             select_tensor(expr, index),
#             select_tensor(compound_state, index),
#             select_tensor(label, index),
#         )
#         self.train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

#     def load_edge_indices(self):
#         """加载边索引"""
#         gene_go = np.load("data/edge_index/gene_go_bp.npy")
#         go_ke = np.load("data/edge_index/go_ke_bp.npy")
#         ke_ke = np.load("data/edge_index/ke_ke.npy")
#         tissue = np.load("data/edge_index/tissue.npy")

#         gene_go = torch.tensor(gene_go, dtype=torch.long, device=self.device)
#         go_ke = torch.tensor(go_ke, dtype=torch.long, device=self.device)
#         ke_ke = torch.tensor(ke_ke, dtype=torch.long, device=self.device)
#         tissue = torch.tensor(tissue, dtype=torch.long, device=self.device)

#         self.edge_indices = (gene_go, go_ke, ke_ke, tissue)

#     def train(self, epochs: int):
#         self.init_dataset()
#         self.load_edge_indices()
#         for epoch in range(epochs):
#             epoch_loss = 0.0
#             for batch in self.train_dataloader:
#                 self.optimizer.zero_grad()
#                 expr, compound, label = [x.to(self.device) for x in batch]
#                 pred, state_pred = self.kan(expr, *self.edge_indices, compound)
#                 loss = self.loss_function(pred, state_pred, label)
#                 loss.backward()
#                 self.optimizer.step()
#                 epoch_loss += loss.item()
#             print(
#                 f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(self.train_dataloader)}"
#             )

#     def calculate_metrics(self, preds: Tensor, labels: Tensor) -> dict[str, float]:
#         pred = torch.sigmoid(preds).detach().cpu().numpy()
#         true = labels.detach().cpu().numpy()
#         pred_binary = (pred > 0.5).astype(int)

#         acc = accuracy_score(true, pred_binary)
#         auc = roc_auc_score(true, pred)
#         precision = precision_score(true, pred_binary)
#         recall = recall_score(true, pred_binary)
#         f1 = f1_score(true, pred_binary)
#         hamming = hamming_loss(true, pred_binary)

#         return {
#             "accuracy": float(acc),
#             "auc": float(auc),
#             "precision": float(precision),
#             "recall": float(recall),
#             "f1": float(f1),
#             "hamming": float(hamming),
#         }

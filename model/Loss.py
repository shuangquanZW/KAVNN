import torch
from torch import nn, Tensor


class AsymmetricLoss(nn.Module):

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss *= one_sided_w
        return -loss


class KAVNNLoss(nn.Module):

    def __init__(self, reduction: str, alpha: float = 0.3) -> None:
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def get_bce_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        if self.reduction == "mean":
            return bce_loss.mean()
        elif self.reduction == "sum":
            return bce_loss.sum()
        else:
            return bce_loss

    def forward(
        self,
        pred: Tensor,
        state: Tensor,
        true: Tensor,
    ) -> Tensor:
        root_loss = self.get_bce_loss(pred, true)
        state_loss = self.get_bce_loss(state, true)
        loss = root_loss + self.alpha * state_loss
        return loss

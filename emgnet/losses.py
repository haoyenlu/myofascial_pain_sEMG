import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseLogisticLoss(nn.Module):
    """
    AUC-oriented ranking loss: mean over all (positive, negative) pairs in the
    batch of softplus(s_neg - s_pos). Operates on raw logits of shape (B,).
    """

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        pos = y_pred[y_true == 1]
        neg = y_pred[y_true == 0]
        if pos.numel() == 0 or neg.numel() == 0:
            return y_pred.sum() * 0.0  # keeps the graph, zero gradient
        return F.softplus(neg.unsqueeze(0) - pos.unsqueeze(1)).mean()


class CombineLoss(nn.Module):
    """Weighted sum of two losses."""

    def __init__(self, loss_1: nn.Module, loss_2: nn.Module, w1: float = 0.5, w2: float = 0.5):
        super().__init__()
        self.loss_1, self.loss_2, self.w1, self.w2 = loss_1, loss_2, w1, w2

    def forward(self, y_pred, y_true):
        return self.w1 * self.loss_1(y_pred, y_true) + self.w2 * self.loss_2(y_pred, y_true)


def get_criterion(name: str, w1: float = 0.5, w2: float = 0.5) -> nn.Module:
    name = name.lower()
    if name == "pairwise":
        return PairwiseLogisticLoss()
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "combine":
        return CombineLoss(PairwiseLogisticLoss(), nn.BCEWithLogitsLoss(), w1, w2)
    raise ValueError(f"Unknown loss '{name}'")

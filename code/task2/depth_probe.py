import torch
import torch.nn as nn
from typing import Tuple

class StructuralDepthProbe(nn.Module):
    """
    Structural Depth Probe from Hewitt & Manning (2019).
    Projects representations via linear map B in R^(k x d).
    Predicted depth is ||B h_i||^2.
    """
    def __init__(self, input_dim: int = 768, probe_rank: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.probe_rank = probe_rank
        self.proj = nn.Linear(input_dim, probe_rank, bias=False)
        nn.init.uniform_(self.proj.weight, -0.05, 0.05)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (N, input_dim) word representations for a sentence
        Returns:
            pred_depth: (N,) predicted word depths
        """
        z = self.proj(h)  # (N, probe_rank)
        pred_depth = torch.sum(z ** 2, dim=-1)  # (N,)
        return pred_depth

    def compute_loss(self, h: torch.Tensor, gold_depths: torch.Tensor) -> torch.Tensor:
        """
        Loss normalized by sentence length (Hewitt & Manning 2019 Eq. 3)
        """
        n = h.shape[0]
        if n == 0:
            return torch.tensor(0.0, device=h.device, requires_grad=True)
        pred_depth = self.forward(h)
        diff = torch.abs(pred_depth - gold_depths)
        loss = torch.sum(diff) / n
        return loss

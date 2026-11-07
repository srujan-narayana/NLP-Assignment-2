import torch
from torch import nn


class StructuralDistanceProbe(nn.Module):
    """Linear projection whose squared distances model dependency-tree distance."""

    def __init__(self, input_dim: int = 768, probe_rank: int = 64) -> None:
        super().__init__()
        if input_dim < 1 or probe_rank < 1 or probe_rank > input_dim:
            raise ValueError("Probe dimensions must satisfy 1 <= rank <= input_dim.")
        self.proj = nn.Linear(input_dim, probe_rank, bias=False)
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        projected = self.proj(hidden)
        squared_norms = projected.square().sum(dim=-1, keepdim=True)
        distances = squared_norms + squared_norms.transpose(-1, -2)
        distances = distances - 2.0 * torch.matmul(projected, projected.transpose(-1, -2))
        distances = distances.clamp_min(0.0)
        distances.diagonal(dim1=-2, dim2=-1).zero_()
        return distances

    def compute_loss(self, hidden: torch.Tensor, gold_distances: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 2 or hidden.shape[0] <= 1:
            return self.proj.weight.sum() * 0.0
        predicted = self(hidden)
        return (predicted - gold_distances).abs().mean()

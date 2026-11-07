import torch
from torch import nn


class StructuralDepthProbe(nn.Module):
    """Linear projection whose squared norms model depth from the dependency root."""

    def __init__(self, input_dim: int = 768, probe_rank: int = 64) -> None:
        super().__init__()
        if input_dim < 1 or probe_rank < 1 or probe_rank > input_dim:
            raise ValueError("Probe dimensions must satisfy 1 <= rank <= input_dim.")
        self.proj = nn.Linear(input_dim, probe_rank, bias=False)
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden).square().sum(dim=-1)

    def compute_loss(self, hidden: torch.Tensor, gold_depths: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 2 or hidden.shape[0] == 0:
            return self.proj.weight.sum() * 0.0
        return (self(hidden) - gold_depths).abs().mean()

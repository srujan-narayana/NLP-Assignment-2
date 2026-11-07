import torch
import torch.nn as nn
from typing import Tuple

class StructuralDistanceProbe(nn.Module):
    """
    Structural Distance Probe from Hewitt & Manning (2019).
    Projects representations via linear map B in R^(k x d).
    Squared distance is d_B(h_i, h_j)^2 = ||B h_i - B h_j||^2.
    """
    def __init__(self, input_dim: int = 768, probe_rank: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.probe_rank = probe_rank
        # Linear layer without bias: x -> x B^T where weight is (rank, input_dim)
        self.proj = nn.Linear(input_dim, probe_rank, bias=False)
        nn.init.uniform_(self.proj.weight, -0.05, 0.05)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (N, input_dim) word representations for a single sentence
        Returns:
            dist_sq: (N, N) pairwise squared distance matrix
        """
        # z: (N, probe_rank)
        z = self.proj(h)
        # Efficient BLAS formulation: ||z_i - z_j||^2 = ||z_i||^2 + ||z_j||^2 - 2 z_i . z_j
        sq_norms = torch.sum(z ** 2, dim=-1, keepdim=True)
        dist_sq = sq_norms + sq_norms.t() - 2.0 * torch.mm(z, z.t())
        dist_sq = torch.clamp(dist_sq, min=0.0)
        dist_sq.fill_diagonal_(0.0)
        return dist_sq

    def compute_loss(self, h: torch.Tensor, gold_distances: torch.Tensor) -> torch.Tensor:
        """
        Loss normalized by sentence length squared (Hewitt & Manning 2019 Eq. 2)
        """
        n = h.shape[0]
        if n <= 1:
            return torch.tensor(0.0, device=h.device, requires_grad=True)
        pred_dist_sq = self.forward(h)
        # L1 difference between predicted squared distance and gold tree distance
        diff = torch.abs(pred_dist_sq - gold_distances)
        loss = torch.sum(diff) / (n * n)
        return loss

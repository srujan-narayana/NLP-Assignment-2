import unittest
import torch
import numpy as np
from task2.distance_probe import StructuralDistanceProbe
from task2.depth_probe import StructuralDepthProbe

class TestProbes(unittest.TestCase):
    def test_distance_probe_forward_and_loss(self):
        input_dim = 768
        rank = 64
        seq_len = 10

        probe = StructuralDistanceProbe(input_dim=input_dim, probe_rank=rank)
        h = torch.randn(seq_len, input_dim)

        dist_sq = probe(h)
        self.assertEqual(dist_sq.shape, (seq_len, seq_len))

        # Check diagonal is 0 (distance to self is 0)
        diag = torch.diag(dist_sq)
        torch.testing.assert_close(diag, torch.zeros(seq_len))

        # Check symmetry: dist_sq[i, j] == dist_sq[j, i]
        torch.testing.assert_close(dist_sq, dist_sq.T)

        # Check non-negativity: ||B h_i - B h_j||^2 >= 0
        self.assertTrue(torch.all(dist_sq >= -1e-6))

        # Check loss computation
        gold_dists = torch.randint(1, 5, (seq_len, seq_len)).float()
        loss = probe.compute_loss(h, gold_dists)
        self.assertTrue(loss.item() >= 0)
        self.assertFalse(torch.isnan(loss))

    def test_depth_probe_forward_and_loss(self):
        input_dim = 768
        rank = 64
        seq_len = 8

        probe = StructuralDepthProbe(input_dim=input_dim, probe_rank=rank)
        h = torch.randn(seq_len, input_dim)

        depths = probe(h)
        self.assertEqual(depths.shape, (seq_len,))

        # Squared norm must be non-negative
        self.assertTrue(torch.all(depths >= 0.0))

        # Check loss computation
        gold_depths = torch.tensor([0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0])
        loss = probe.compute_loss(h, gold_depths)
        self.assertTrue(loss.item() >= 0)
        self.assertFalse(torch.isnan(loss))

if __name__ == "__main__":
    unittest.main()

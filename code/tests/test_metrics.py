import unittest
import numpy as np
from common.metrics import (
    compute_cosine_similarity,
    compute_spearman_correlation,
    extract_gold_undirected_edges,
    compute_mst_undirected_edges,
    compute_uuas,
    solve_analogy_3cosadd
)

class TestMetrics(unittest.TestCase):
    def test_cosine_similarity(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0])
        v3 = np.array([0.0, 1.0, 0.0])
        v_zero = np.array([0.0, 0.0, 0.0])

        self.assertAlmostEqual(compute_cosine_similarity(v1, v2), 1.0)
        self.assertAlmostEqual(compute_cosine_similarity(v1, v3), 0.0)
        self.assertAlmostEqual(compute_cosine_similarity(v1, v_zero), 0.0)

    def test_spearman_correlation(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        corr, _ = compute_spearman_correlation(x, y)
        self.assertAlmostEqual(corr, 1.0)

        y_inv = [10, 8, 6, 4, 2]
        corr_inv, _ = compute_spearman_correlation(x, y_inv)
        self.assertAlmostEqual(corr_inv, -1.0)

    def test_mst_and_uuas(self):
        # Line graph 0 - 1 - 2 - 3
        dist_mat = np.array([
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 1.0],
            [3.0, 2.0, 1.0, 0.0]
        ])
        mst_edges = compute_mst_undirected_edges(dist_mat)
        expected_edges = {(0, 1), (1, 2), (2, 3)}
        self.assertEqual(mst_edges, expected_edges)

        # Gold heads: 0 is root (head=0), 1 -> 0 (head=1), 2 -> 1 (head=2), 3 -> 2 (head=3)
        heads = [0, 1, 2, 3]
        gold_edges = extract_gold_undirected_edges(heads)
        self.assertEqual(gold_edges, expected_edges)

        uuas, correct, total = compute_uuas(mst_edges, gold_edges)
        self.assertEqual(uuas, 1.0)
        self.assertEqual(correct, 3)
        self.assertEqual(total, 3)

    def test_analogy_3cosadd(self):
        # man : king :: woman : queen
        vocab = ["man", "woman", "king", "queen", "apple"]
        word_to_idx = {w: i for i, w in enumerate(vocab)}

        # Construct synthetic vectors: dim 2
        # gender on axis 0: man=-1, woman=+1
        # royalty on axis 1: common=0, royal=+1
        matrix = np.array([
            [-1.0, 0.0],  # man
            [+1.0, 0.0],  # woman
            [-1.0, 1.0],  # king
            [+1.0, 1.0],  # queen
            [0.0, -1.0],  # apple
        ], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix_normed = matrix / norms

        pred = solve_analogy_3cosadd(
            matrix[word_to_idx["man"]],
            matrix[word_to_idx["king"]],
            matrix[word_to_idx["woman"]],
            vocab,
            matrix_normed,
            word_to_idx,
            exclude_words={"man", "king", "woman"}
        )
        self.assertEqual(pred, "queen")

if __name__ == "__main__":
    unittest.main()

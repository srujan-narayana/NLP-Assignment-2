import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.metrics import compute_mst_undirected_edges
from task1.cooccurrence import build_cooccurrence_matrix
from task1.dataset import build_vocabulary
from task1.evaluation import evaluate_google_analogy, evaluate_wordsim353
from task1.ppmi import compute_ppmi_matrix
from task1.svd_embeddings import compute_svd_embeddings


class Task1Tests(unittest.TestCase):
    def test_symmetric_window_and_sentence_boundaries(self):
        vocabulary = {"a": 0, "b": 1, "c": 2}
        matrix = build_cooccurrence_matrix(
            [["a", "b"], ["c"]], vocabulary, window_size=1
        ).toarray()
        np.testing.assert_array_equal(matrix, matrix.T)
        self.assertEqual(matrix[0, 1], 1.0)
        self.assertEqual(matrix[0, 2], 0.0)

    def test_oov_tokens_keep_their_original_window_positions(self):
        matrix = build_cooccurrence_matrix(
            [["a", "filtered", "b"]], {"a": 0, "b": 1}, window_size=1
        )
        self.assertEqual(matrix.nnz, 0)

    def test_window_five_and_oov_filtering(self):
        matrix = build_cooccurrence_matrix(
            [["a", "x", "x", "x", "x", "b"]],
            {"a": 0, "b": 1},
            window_size=5,
        )
        self.assertEqual(matrix[0, 1], 1.0)
        self.assertEqual(matrix[1, 0], 1.0)

    def test_ppmi_values_and_empty_matrix(self):
        matrix = sp.csr_matrix([[0.0, 2.0], [2.0, 0.0]])
        ppmi = compute_ppmi_matrix(matrix)
        np.testing.assert_allclose(ppmi.toarray(), [[0.0, 1.0], [1.0, 0.0]])
        self.assertEqual(compute_ppmi_matrix(sp.csr_matrix((3, 3))).nnz, 0)

    def test_svd_handles_small_and_empty_vocabularies(self):
        self.assertEqual(compute_svd_embeddings(sp.csr_matrix((0, 0))).shape, (0, 100))
        self.assertEqual(compute_svd_embeddings(sp.csr_matrix((1, 1))).shape, (1, 100))
        vectors = compute_svd_embeddings(
            compute_ppmi_matrix(sp.csr_matrix([[0.0, 2.0], [2.0, 0.0]])),
            dim=5,
        )
        self.assertEqual(vectors.shape, (2, 5))
        self.assertTrue(np.isfinite(vectors).all())

    def test_vocabulary_is_deterministic(self):
        word_to_idx, words, counts = build_vocabulary(
            [["b", "a", "a"], ["c", "b"]], min_freq=1
        )
        self.assertEqual(words, ["a", "b", "c"])
        self.assertEqual(word_to_idx, {"a": 0, "b": 1, "c": 2})
        self.assertEqual(counts["a"], 2)

    def test_wordsim_reports_oov_coverage(self):
        vectors = {
            "near1": np.array([1.0, 0.0]),
            "near2": np.array([0.9, 0.1]),
            "far": np.array([0.0, 1.0]),
        }
        result = evaluate_wordsim353(
            vectors, [("near1", "near2", 9.0), ("near1", "far", 1.0), ("x", "far", 2.0)]
        )
        self.assertEqual(result["total_pairs"], 3)
        self.assertEqual(result["evaluated_count"], 2)
        self.assertEqual(result["oov_count"], 1)

    def test_analogy_uses_normalized_query_vectors(self):
        vectors = {
            "a": np.array([-1.0, 0.0]),
            "b": np.array([-100.0, 100.0]),
            "c": np.array([1.0, 0.0]),
            "answer": np.array([1.0, 1.0]),
            "decoy": np.array([-1.0, 1.0]),
        }
        result = evaluate_google_analogy(
            vectors, [("test", "a", "b", "c", "answer")]
        )
        self.assertEqual(result["correct"], 1)

    def test_mst_has_n_minus_one_edges_and_stable_ties(self):
        distances = np.ones((4, 4), dtype=np.float32)
        np.fill_diagonal(distances, 0.0)
        first = compute_mst_undirected_edges(distances)
        second = compute_mst_undirected_edges(distances)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)


if __name__ == "__main__":
    unittest.main()

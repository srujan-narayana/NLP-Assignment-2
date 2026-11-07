import unittest
import numpy as np
import scipy.sparse as sp
from task1.cooccurrence import build_cooccurrence_matrix
from task1.ppmi import compute_ppmi_matrix
from task1.svd_embeddings import compute_svd_embeddings

class TestEmbeddings(unittest.TestCase):
    def setUp(self):
        self.sentences = [
            ["natural", "language", "processing", "with", "deep", "learning"],
            ["deep", "learning", "for", "natural", "language", "understanding"],
            ["statistical", "semantics", "and", "distributional", "representations"]
        ]
        all_words = set(w for s in self.sentences for w in s)
        self.word_to_idx = {w: i for i, w in enumerate(sorted(all_words))}
        self.idx_to_word = sorted(all_words)

    def test_cooccurrence_symmetry(self):
        cooc = build_cooccurrence_matrix(self.sentences, self.word_to_idx, window_size=2)
        dense = cooc.toarray()
        # Co-occurrence within symmetric window should be symmetric
        np.testing.assert_allclose(dense, dense.T)
        # Diagonal should be 0 (words don't co-occur with themselves at step != 0)
        self.assertTrue(np.all(np.diag(dense) == 0))

    def test_ppmi_non_negativity(self):
        cooc = build_cooccurrence_matrix(self.sentences, self.word_to_idx, window_size=2)
        ppmi = compute_ppmi_matrix(cooc)
        self.assertTrue(np.all(ppmi.data >= 0.0))
        # Ensure no NaN or infinite values
        self.assertFalse(np.any(np.isnan(ppmi.data)))
        self.assertFalse(np.any(np.isinf(ppmi.data)))

    def test_svd_dimension(self):
        cooc = build_cooccurrence_matrix(self.sentences, self.word_to_idx, window_size=2)
        ppmi = compute_ppmi_matrix(cooc)
        emb = compute_svd_embeddings(ppmi, dim=10, random_state=42)
        self.assertEqual(emb.shape, (len(self.word_to_idx), 10))
        self.assertFalse(np.any(np.isnan(emb)))

if __name__ == "__main__":
    unittest.main()

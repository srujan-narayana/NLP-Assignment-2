import unittest
from task1.dataset import build_vocabulary

class TestPreprocessing(unittest.TestCase):
    def test_vocabulary_construction(self):
        sample_sentences = [
            ["the", "quick", "brown", "fox", "jumps"],
            ["the", "lazy", "dog", "the", "fox"]
        ]
        word_to_idx, idx_to_word, counts = build_vocabulary(sample_sentences, min_freq=1)
        
        self.assertIn("the", word_to_idx)
        self.assertEqual(counts["the"], 3)
        self.assertEqual(counts["fox"], 2)
        self.assertEqual(len(word_to_idx), len(idx_to_word))
        # Ensure indexing is zero-based and consecutive
        for idx, word in enumerate(idx_to_word):
            self.assertEqual(word_to_idx[word], idx)

    def test_min_frequency_filtering(self):
        sample_sentences = [
            ["apple", "banana", "apple"],
            ["cherry", "banana", "apple"]
        ]
        word_to_idx, idx_to_word, counts = build_vocabulary(sample_sentences, min_freq=2)
        self.assertIn("apple", word_to_idx)
        self.assertIn("banana", word_to_idx)
        self.assertNotIn("cherry", word_to_idx)
        self.assertEqual(len(word_to_idx), 2)

if __name__ == "__main__":
    unittest.main()

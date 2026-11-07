import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.data_utils import UDSentence, load_ud_gum_dataset, parse_conllu
from task2.alignment import align_words_to_subwords, pool_word_representations
from task2.bert_extractor import BertLayerExtractor
from task2.depth_probe import StructuralDepthProbe
from task2.distance_probe import StructuralDistanceProbe
from task2.evaluation import evaluate_depth_probe_correlation
from task2.evaluation import evaluate_distance_probe_uuas
from task2.gold_structures import build_gold_structures
from task2.main import train_distance_probe, train_depth_probe
from test_support import temporary_directory


class DummyTokenizer:
    cls_token_id = 101
    sep_token_id = 102
    unk_token_id = 100

    def encode(self, word, add_special_tokens=False):
        return {"a": [5], "long": [6, 7, 8], "b": [9]}.get(word, [10])


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=2, num_hidden_layers=2, _commit_hash="dummy-revision"
        )
        self.calls = 0

    def forward(self, input_ids, output_hidden_states=True):
        self.calls += 1
        base = input_ids.float().unsqueeze(-1).repeat(1, 1, 2)
        states = (base, base + 1.0, base + 2.0)
        return SimpleNamespace(hidden_states=states)


class DataAndProbeTests(unittest.TestCase):
    def test_conllu_parser_skips_ranges_and_empty_nodes(self):
        content = (
            "# sent_id = 1\n"
            "1-2\tcan't\t_\t_\t_\t_\t_\t_\t_\t_\n"
            "1\tca\t_\tAUX\t_\t_\t2\taux\t_\t_\n"
            "2\tn't\t_\tPART\t_\t_\t0\troot\t_\t_\n"
            "2.1\tghost\t_\tX\t_\t_\t_\t_\t_\t_\n\n"
        )
        with temporary_directory() as folder:
            path = Path(folder) / "sample.conllu"
            path.write_text(content, encoding="utf-8")
            sentence = parse_conllu(path)[0]
        self.assertEqual(sentence.tokens, ("ca", "n't"))
        self.assertEqual(sentence.heads, (2, 0))

    def test_local_only_dataset_mode_fails_without_network_fallback(self):
        with temporary_directory() as folder:
            with self.assertRaisesRegex(FileNotFoundError, "Local-only mode"):
                load_ud_gum_dataset(folder, local_files_only=True)

    def test_gold_distances_and_depths(self):
        distances, depths = build_gold_structures([0, 1, 2])
        np.testing.assert_array_equal(distances, [[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        np.testing.assert_array_equal(depths, [0, 1, 2])

    def test_invalid_gold_tree_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly one root"):
            build_gold_structures([0, 0])
        with self.assertRaisesRegex(ValueError, "connected tree"):
            build_gold_structures([0, 3, 2])
        with self.assertRaisesRegex(ValueError, "own dependency head"):
            build_gold_structures([1])

    def test_wordpiece_alignment_and_pooling(self):
        ids, indices = align_words_to_subwords(["long", "a"], DummyTokenizer())
        self.assertEqual(ids, [101, 6, 7, 8, 5, 102])
        hidden = torch.arange(len(ids), dtype=torch.float32).unsqueeze(1)
        pooled = pool_word_representations(hidden, indices)
        torch.testing.assert_close(pooled[:, 0], torch.tensor([2.0, 4.0]))
        with self.assertRaises(ValueError):
            pool_word_representations(hidden, indices, strategy="bad")

    def test_extractor_chunks_without_losing_or_misaligning_words(self):
        model = DummyModel()
        extractor = BertLayerExtractor(
            model_name="dummy",
            device="cpu",
            tokenizer=DummyTokenizer(),
            model=model,
            max_tokens=4,
        )
        sentences = [UDSentence(("a", "long", "b"), (2, 0, 2))]
        with temporary_directory() as folder:
            store = extractor.ensure_features("train", sentences, [1, 2], folder)
            layer1 = store.layer(1)[0]
            calls = model.calls
            cached = extractor.ensure_features("train", sentences, [1, 2], folder)
            self.assertEqual(model.calls, calls)
            torch.testing.assert_close(layer1[:, 0], torch.tensor([6.0, 8.0, 10.0]))
            torch.testing.assert_close(cached.layer(2)[0][:, 0], torch.tensor([7.0, 9.0, 11.0]))

    def test_probe_forward_losses_and_training(self):
        torch.manual_seed(4)
        sentences = [
            UDSentence(("a", "b", "c"), (0, 1, 2)),
            UDSentence(("d", "e"), (2, 0)),
        ]
        reps = [torch.randn(3, 4), torch.randn(2, 4)]
        distances = [build_gold_structures(sentence.heads)[0] for sentence in sentences]
        depths = [build_gold_structures(sentence.heads)[1] for sentence in sentences]
        distance_probe = StructuralDistanceProbe(input_dim=4, probe_rank=2)
        depth_probe = StructuralDepthProbe(input_dim=4, probe_rank=2)
        distance = distance_probe(reps[0])
        torch.testing.assert_close(distance, distance.T)
        self.assertTrue(torch.all(distance.diag() == 0.0))
        self.assertGreaterEqual(
            float(distance_probe.compute_loss(reps[0], torch.tensor(distances[0])).detach()),
            0.0,
        )
        self.assertGreaterEqual(
            float(depth_probe.compute_loss(reps[0], torch.tensor(depths[0])).detach()),
            0.0,
        )

        trained_distance, _ = train_distance_probe(
            reps, distances, reps, sentences, rank=2, epochs=1, batch_size=2
        )
        trained_depth, _ = train_depth_probe(
            reps, depths, reps, depths, rank=2, epochs=1, batch_size=2
        )
        metrics = evaluate_distance_probe_uuas(trained_distance, reps, sentences)
        correlations = evaluate_depth_probe_correlation(trained_depth, reps, depths)
        self.assertEqual(metrics["gold_edges"], 3)
        self.assertEqual(correlations["evaluated_tokens"], 5)


if __name__ == "__main__":
    unittest.main()

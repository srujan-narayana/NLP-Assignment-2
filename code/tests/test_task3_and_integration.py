import csv
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.data_utils import UDSentence
from task2.main import run_task2
from task3.static_vectors import construct_static_embeddings_from_bert
from task3.main import run_task3
from test_support import temporary_directory


class FakeFeatureStore:
    hidden_size = 4

    def __init__(self, representations):
        self.representations = representations

    def layer(self, layer_number):
        return [value.clone() for value in self.representations]


class FakeExtractor:
    device = torch.device("cpu")

    def __init__(self, by_split):
        self.by_split = by_split

    def ensure_features(self, split_name, sentences, layers, cache_dir):
        return FakeFeatureStore(self.by_split[split_name])


class Task3AndIntegrationTests(unittest.TestCase):
    def test_static_pooling_uses_online_mean_and_checks_alignment(self):
        sentences = [
            UDSentence(("Word", "other"), (0, 1)),
            UDSentence(("word",), (0,)),
        ]
        vectors = [
            torch.tensor([[1.0, 3.0], [10.0, 0.0]]),
            torch.tensor([[5.0, 7.0]]),
        ]
        embeddings, counts = construct_static_embeddings_from_bert(sentences, vectors)
        np.testing.assert_allclose(embeddings["word"], [3.0, 5.0])
        self.assertEqual(counts["word"], 2)
        with self.assertRaises(ValueError):
            construct_static_embeddings_from_bert(sentences, vectors[:1])

    def test_task2_and_task3_runner_integration_with_local_features(self):
        torch.manual_seed(11)
        splits = {
            "train": [
                UDSentence(("alpha", "beta", "gamma"), (0, 1, 2)),
                UDSentence(("beta", "gamma"), (2, 0)),
            ],
            "validation": [
                UDSentence(("alpha", "delta", "beta"), (2, 0, 2)),
                UDSentence(("gamma", "alpha"), (2, 0)),
            ],
            "test": [
                UDSentence(("alpha", "beta", "delta"), (2, 0, 2)),
                UDSentence(("gamma", "beta"), (2, 0)),
            ],
        }
        features = {
            split: [
                torch.randn(len(sentence.tokens), 4) for sentence in sentences
            ]
            for split, sentences in splits.items()
        }
        extractor = FakeExtractor(features)
        with temporary_directory() as folder:
            root = Path(folder)
            selected_layer, task2_result = run_task2(
                splits=splits,
                extractor=extractor,
                layers=[1],
                ranks=[2],
                epochs=1,
                batch_size=2,
                output_dir=root / "results" / "task2",
                checkpoints_dir=root / "checkpoints" / "task2",
                figures_dir=root / "results" / "figures",
                cache_dir=root / "cache",
            )
            self.assertEqual(selected_layer, 1)
            self.assertEqual(task2_result["results_by_rank"]["2"]["1"]["gold_edges"], 3)

            task1_results = root / "results" / "task1" / "task1_results.json"
            task1_results.parent.mkdir(parents=True)
            task1_results.write_text(
                json.dumps(
                    {
                        "wordsim353": {"spearman_correlation": 0.1},
                        "google_analogy": {"accuracy": 0.2},
                        "structural_check": {"overall_attachment_agreement": 0.3},
                    }
                ),
                encoding="utf-8",
            )
            task3_result = run_task3(
                splits=splits,
                wordsim_pairs=[("alpha", "beta", 7.0), ("alpha", "gamma", 2.0)],
                analogy_questions=[("test", "alpha", "beta", "gamma", "delta")],
                selected_layer=selected_layer,
                extractor=extractor,
                output_dir=root / "results" / "task3",
                checkpoints_dir=root / "checkpoints" / "task3",
                task1_results_path=task1_results,
                task2_results_path=root / "results" / "task2" / "task2_results.json",
                sample_indices_path=root / "data" / "sample.json",
                cache_dir=root / "cache",
            )
            self.assertEqual(task3_result["selected_layer"], 1)
            self.assertEqual(len(task3_result["comparison"]), 3)
            self.assertTrue((root / "results" / "task3" / "comparison_table.csv").exists())

    def test_task1_command_line_smoke_run(self):
        with temporary_directory() as folder:
            root = Path(folder)
            data_root = root / "data"
            gum = data_root / "ud_gum"
            gum.mkdir(parents=True)
            conllu = (
                "1\tking\t_\tNOUN\t_\t_\t2\tnsubj\t_\t_\n"
                "2\tis\t_\tAUX\t_\t_\t0\troot\t_\t_\n"
                "3\tman\t_\tNOUN\t_\t_\t2\tobj\t_\t_\n\n"
                "1\tqueen\t_\tNOUN\t_\t_\t2\tnsubj\t_\t_\n"
                "2\tis\t_\tAUX\t_\t_\t0\troot\t_\t_\n"
                "3\twoman\t_\tNOUN\t_\t_\t2\tobj\t_\t_\n\n"
                "1\tking\t_\tNOUN\t_\t_\t2\tnsubj\t_\t_\n"
                "2\tand\t_\tCCONJ\t_\t_\t0\troot\t_\t_\n"
                "3\tqueen\t_\tNOUN\t_\t_\t2\tconj\t_\t_\n\n"
            )
            for split in ("train", "validation", "test"):
                (gum / f"{split}.conllu").write_text(conllu, encoding="utf-8")
            with (data_root / "wordsim353.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.writer(stream)
                writer.writerow(["word1", "word2", "similarity"])
                writer.writerows(
                    [("king", "queen", 9.0), ("king", "woman", 3.0), ("is", "man", 2.0)]
                )
            (data_root / "questions-words.txt").write_text(
                ": synthetic\nking queen man woman\n", encoding="utf-8"
            )
            output = root / "results"
            checkpoints = root / "checkpoints"
            command = [
                sys.executable,
                str(Path(__file__).resolve().parents[1] / "main.py"),
                "--task",
                "1",
                "--data-dir",
                str(data_root),
                "--output-dir",
                str(output),
                "--checkpoints-dir",
                str(checkpoints),
            ]
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[2],
                text=True,
                capture_output=True,
                timeout=60,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            result_path = output / "task1" / "task1_results.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["window_size"], 5)
            self.assertEqual(result["structural_check"]["num_sentences"], 3)
            self.assertTrue((checkpoints / "task1" / "svd_embeddings.npy").exists())


if __name__ == "__main__":
    unittest.main()

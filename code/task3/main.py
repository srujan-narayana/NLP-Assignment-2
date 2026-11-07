import csv
import json
import logging
import pickle
from pathlib import Path

import numpy as np

from common.data_utils import UDSentence, load_google_analogies, load_wordsim353_dataset
from common.data_utils import load_ud_gum_dataset
from common.reproducibility import set_seed
from config import (
    BERT_MODEL_NAME,
    CACHE_DIR,
    CHECKPOINTS_DIR,
    DATA_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    STRUCTURAL_SAMPLE_SEED,
    STRUCTURAL_SAMPLE_SIZE,
)
from task1.structural_check import evaluate_structural_mst
from task2.bert_extractor import BertLayerExtractor
from task3.evaluation import evaluate_bert_static_embeddings
from task3.static_vectors import construct_static_embeddings_from_bert

logger = logging.getLogger(__name__)


def _read_selected_layer(task2_results: Path) -> int:
    if not task2_results.exists():
        raise FileNotFoundError(
            "Task 3 needs Task 2's selected layer. Run Task 2 first or pass selected_layer."
        )
    summary = json.loads(task2_results.read_text(encoding="utf-8"))
    layer = int(summary["selected_layer"])
    if not 1 <= layer <= 12:
        raise ValueError(f"Task 2 selected an invalid BERT layer: {layer}.")
    return layer


def _write_comparison(
    task1_results_path: Path,
    task3_metrics: dict[str, object],
    output_path: Path,
) -> list[dict[str, object]]:
    if not task1_results_path.exists():
        raise FileNotFoundError(
            "Task 3 needs Task 1 results for the required comparison. Run Task 1 first."
        )
    baseline = json.loads(task1_results_path.read_text(encoding="utf-8"))
    task1_values = {
        "WordSim-353 Spearman": baseline["wordsim353"]["spearman_correlation"],
        "Google analogy accuracy": baseline["google_analogy"]["accuracy"],
        "MST attachment agreement": baseline["structural_check"][
            "overall_attachment_agreement"
        ],
    }
    task3_values = {
        "WordSim-353 Spearman": task3_metrics["wordsim353"]["spearman_correlation"],
        "Google analogy accuracy": task3_metrics["google_analogy"]["accuracy"],
        "MST attachment agreement": task3_metrics["structural_check"][
            "overall_attachment_agreement"
        ],
    }
    rows: list[dict[str, object]] = []
    for metric, task3_value in task3_values.items():
        task1_value = float(task1_values[metric])
        task3_value = float(task3_value)
        rows.append(
            {
                "metric": metric,
                "task1": task1_value,
                "task3": task3_value,
                "delta_task3_minus_task1": task3_value - task1_value,
            }
        )
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["metric", "task1", "task3", "delta_task3_minus_task1"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def run_task3(
    splits: dict[str, list[UDSentence]] | None = None,
    wordsim_pairs: list[tuple[str, str, float]] | None = None,
    analogy_questions: list[tuple[str, str, str, str, str]] | None = None,
    selected_layer: int | None = None,
    extractor: BertLayerExtractor | None = None,
    output_dir: str | Path | None = None,
    checkpoints_dir: str | Path | None = None,
    task1_results_path: str | Path | None = None,
    task2_results_path: str | Path | None = None,
    sample_indices_path: str | Path | None = None,
    cache_dir: str | Path = CACHE_DIR,
    local_files_only: bool = False,
    model_name: str = BERT_MODEL_NAME,
    device: str | None = None,
) -> dict[str, object]:
    set_seed(RANDOM_SEED)
    splits = splits or load_ud_gum_dataset(local_files_only=local_files_only)
    if not splits.get("train") or not splits.get("test"):
        raise ValueError("Task 3 requires non-empty train and test splits.")
    wordsim_pairs = wordsim_pairs if wordsim_pairs is not None else load_wordsim353_dataset()
    analogy_questions = (
        analogy_questions if analogy_questions is not None else load_google_analogies()
    )
    result_path = Path(output_dir) if output_dir is not None else RESULTS_DIR / "task3"
    checkpoint_path = (
        Path(checkpoints_dir) if checkpoints_dir is not None else CHECKPOINTS_DIR / "task3"
    )
    result_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    comparison_path = (
        Path(task1_results_path)
        if task1_results_path is not None
        else RESULTS_DIR / "task1" / "task1_results.json"
    )
    if not comparison_path.exists():
        raise FileNotFoundError(
            "Task 3 needs Task 1 results for the required comparison. Run Task 1 first."
        )
    layer_info_path = (
        Path(task2_results_path)
        if task2_results_path is not None
        else RESULTS_DIR / "task2" / "task2_results.json"
    )
    layer = selected_layer if selected_layer is not None else _read_selected_layer(layer_info_path)
    if not 1 <= int(layer) <= 12:
        raise ValueError("selected_layer must be in 1..12.")
    layer = int(layer)

    extractor = extractor or BertLayerExtractor(
        model_name=model_name, device=device, local_files_only=local_files_only
    )
    store = extractor.ensure_features("train", splits["train"], [layer], cache_dir)
    representations = store.layer(layer)
    embeddings, frequencies = construct_static_embeddings_from_bert(
        splits["train"], representations
    )
    with (checkpoint_path / f"bert_static_embeddings_layer_{layer}.pkl").open("wb") as stream:
        pickle.dump(embeddings, stream, protocol=pickle.HIGHEST_PROTOCOL)
    (checkpoint_path / "word_frequencies.json").write_text(
        json.dumps(frequencies, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    semantic = evaluate_bert_static_embeddings(
        embeddings, wordsim_pairs, analogy_questions
    )
    sample_path = (
        Path(sample_indices_path)
        if sample_indices_path is not None
        else DATA_DIR / "structural_sample_indices.json"
    )
    structural = evaluate_structural_mst(
        embeddings,
        splits["test"],
        sample_path=sample_path,
        sample_size=STRUCTURAL_SAMPLE_SIZE,
        seed=STRUCTURAL_SAMPLE_SEED,
    )
    metrics: dict[str, object] = {
        **semantic,
        "structural_check": structural,
    }
    comparison = _write_comparison(
        comparison_path, metrics, result_path / "comparison_table.csv"
    )
    summary: dict[str, object] = {
        "task": "Task 3: BERT-derived static vectors",
        "dataset": "UD English GUM",
        "selected_layer": layer,
        "vocabulary_size": len(embeddings),
        "wordsim353": semantic["wordsim353"],
        "google_analogy": semantic["google_analogy"],
        "structural_check": structural,
        "comparison": comparison,
    }
    (result_path / "task3_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.info(
        "Task 3 complete from layer %d: vocabulary=%d.",
        layer,
        len(embeddings),
    )
    return summary

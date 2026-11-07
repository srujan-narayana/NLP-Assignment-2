import json
import logging
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from common.data_utils import UDSentence, load_google_analogies, load_wordsim353_dataset
from common.data_utils import load_ud_gum_dataset
from common.reproducibility import set_seed
from config import (
    CHECKPOINTS_DIR,
    DATA_DIR,
    EMBEDDING_DIM,
    MIN_WORD_FREQ,
    RANDOM_SEED,
    RESULTS_DIR,
    STRUCTURAL_SAMPLE_SEED,
    STRUCTURAL_SAMPLE_SIZE,
    WINDOW_SIZE,
)
from task1.cooccurrence import build_cooccurrence_matrix
from task1.dataset import build_vocabulary
from task1.evaluation import evaluate_google_analogy, evaluate_wordsim353
from task1.ppmi import compute_ppmi_matrix
from task1.structural_check import evaluate_structural_mst
from task1.svd_embeddings import build_embedding_dict, compute_svd_embeddings

logger = logging.getLogger(__name__)


def run_task1(
    splits: dict[str, list[UDSentence]] | None = None,
    wordsim_pairs: list[tuple[str, str, float]] | None = None,
    analogy_questions: list[tuple[str, str, str, str, str]] | None = None,
    output_dir: str | Path | None = None,
    checkpoints_dir: str | Path | None = None,
    sample_indices_path: str | Path | None = None,
) -> dict[str, object]:
    set_seed(RANDOM_SEED)
    splits = splits or load_ud_gum_dataset()
    if not {"train", "test"}.issubset(splits):
        raise ValueError("Task 1 requires train and test splits.")
    wordsim_pairs = wordsim_pairs if wordsim_pairs is not None else load_wordsim353_dataset()
    analogy_questions = (
        analogy_questions if analogy_questions is not None else load_google_analogies()
    )
    result_path = Path(output_dir) if output_dir is not None else RESULTS_DIR / "task1"
    checkpoint_path = (
        Path(checkpoints_dir) if checkpoints_dir is not None else CHECKPOINTS_DIR / "task1"
    )
    sample_path = (
        Path(sample_indices_path)
        if sample_indices_path is not None
        else DATA_DIR / "structural_sample_indices.json"
    )
    result_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    sentences = [sentence.tokens for sentence in splits["train"]]
    word_to_idx, vocabulary, counts = build_vocabulary(sentences, MIN_WORD_FREQ)
    cooccurrence = build_cooccurrence_matrix(sentences, word_to_idx, WINDOW_SIZE)
    ppmi = compute_ppmi_matrix(cooccurrence)
    embeddings = compute_svd_embeddings(ppmi, EMBEDDING_DIM, RANDOM_SEED)
    embedding_dict = build_embedding_dict(embeddings, vocabulary)

    sp.save_npz(checkpoint_path / "cooccurrence_matrix.npz", cooccurrence)
    sp.save_npz(checkpoint_path / "ppmi_matrix.npz", ppmi)
    np.save(checkpoint_path / "svd_embeddings.npy", embeddings)
    (checkpoint_path / "vocab.json").write_text(
        json.dumps(
            {
                "word_to_idx": word_to_idx,
                "idx_to_word": vocabulary,
                "counts": counts,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    wordsim_results = evaluate_wordsim353(embedding_dict, wordsim_pairs)
    analogy_results = evaluate_google_analogy(embedding_dict, analogy_questions)
    structural_results = evaluate_structural_mst(
        embedding_dict,
        splits["test"],
        sample_path=sample_path,
        sample_size=STRUCTURAL_SAMPLE_SIZE,
        seed=STRUCTURAL_SAMPLE_SEED,
    )
    summary: dict[str, object] = {
        "task": "Task 1: PPMI + SVD static vectors",
        "dataset": "UD English GUM",
        "vocabulary_size": len(vocabulary),
        "embedding_dimension": EMBEDDING_DIM,
        "window_size": WINDOW_SIZE,
        "minimum_word_frequency": MIN_WORD_FREQ,
        "wordsim353": wordsim_results,
        "google_analogy": analogy_results,
        "structural_check": structural_results,
    }
    (result_path / "task1_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.info(
        "Task 1 complete: vocabulary=%d, WordSim rho=%.4f, analogy accuracy=%.4f.",
        len(vocabulary),
        wordsim_results["spearman_correlation"],
        analogy_results["accuracy"],
    )
    return summary

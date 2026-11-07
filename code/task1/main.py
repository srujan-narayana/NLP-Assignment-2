import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
import json
import pickle
import numpy as np
import scipy.sparse as sp

from config import RESULTS_DIR, CHECKPOINTS_DIR, RANDOM_SEED, EMBEDDING_DIM
from common.reproducibility import set_seed
from task1.dataset import get_tokenized_corpus, build_vocabulary
from task1.cooccurrence import build_cooccurrence_matrix
from task1.ppmi import compute_ppmi_matrix
from task1.svd_embeddings import compute_svd_embeddings, build_embedding_dict
from task1.evaluation import evaluate_wordsim353, evaluate_google_analogy
from task1.structural_check import evaluate_structural_mst

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Task1")

def run_task1():
    set_seed(RANDOM_SEED)
    output_dir = RESULTS_DIR / "task1"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = CHECKPOINTS_DIR / "task1"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Starting Task 1: PPMI + SVD Static Word Vectors ===")

    # 1. Dataset & Vocab
    sentences = get_tokenized_corpus("train")
    word_to_idx, idx_to_word, counts = build_vocabulary(sentences)

    # Save vocab
    with open(ckpt_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump({"word_to_idx": word_to_idx, "idx_to_word": idx_to_word}, f)

    # 2. Co-occurrence Matrix
    cooc_matrix = build_cooccurrence_matrix(sentences, word_to_idx)
    sp.save_npz(ckpt_dir / "cooccurrence_matrix.npz", cooc_matrix)

    # 3. PPMI Matrix
    ppmi_matrix = compute_ppmi_matrix(cooc_matrix)
    sp.save_npz(ckpt_dir / "ppmi_matrix.npz", ppmi_matrix)

    # 4. Truncated SVD (100 dims)
    embeddings = compute_svd_embeddings(ppmi_matrix, dim=EMBEDDING_DIM, random_state=RANDOM_SEED)
    embeddings_dict = build_embedding_dict(embeddings, idx_to_word)
    np.save(ckpt_dir / "svd_embeddings.npy", embeddings)
    with open(ckpt_dir / "embeddings_dict.pkl", "wb") as f:
        pickle.dump(embeddings_dict, f)

    # 5. WordSim-353
    logger.info("Evaluating WordSim-353...")
    wordsim_res = evaluate_wordsim353(embeddings_dict)

    # 6. Google Analogy
    logger.info("Evaluating Google Analogy...")
    analogy_res = evaluate_google_analogy(embeddings_dict)

    # 7. Bonus Structural Check
    logger.info("Evaluating Bonus Structural MST Check...")
    structural_res = evaluate_structural_mst(embeddings_dict, default_dim=EMBEDDING_DIM)

    # Aggregate results
    summary = {
        "task": "Task 1: PPMI + SVD Static Vectors",
        "vocabulary_size": len(word_to_idx),
        "embedding_dimension": EMBEDDING_DIM,
        "wordsim353": wordsim_res,
        "google_analogy": {
            "accuracy": analogy_res["accuracy"],
            "correct": analogy_res["correct"],
            "evaluated": analogy_res["evaluated"],
            "dropped": analogy_res["dropped"],
            "total_questions": analogy_res["total_questions"]
        },
        "structural_check": {
            "overall_attachment_agreement": structural_res["overall_attachment_agreement"],
            "macro_attachment_agreement": structural_res["macro_attachment_agreement"],
            "total_correct_edges": structural_res["total_correct_edges"],
            "total_gold_edges": structural_res["total_gold_edges"],
            "num_sentences": structural_res["num_sentences"]
        }
    }

    with open(output_dir / "task1_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("=== Task 1 Completed Successfully! ===")
    return summary

if __name__ == "__main__":
    run_task1()

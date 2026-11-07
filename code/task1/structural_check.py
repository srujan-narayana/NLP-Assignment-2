import logging
import random
import json
import numpy as np
from typing import Dict, List, Tuple, Any
from config import STRUCTURAL_SAMPLE_SIZE, STRUCTURAL_SAMPLE_SEED, DATA_DIR
from common.data_utils import load_ud_gum_dataset
from common.metrics import (
    compute_cosine_similarity,
    compute_mst_undirected_edges,
    extract_gold_undirected_edges,
    compute_uuas
)

logger = logging.getLogger(__name__)

INDICES_FILE = DATA_DIR / "structural_check_test_indices.json"

def get_structural_check_indices(total_test_sentences: int) -> List[int]:
    """
    Deterministically sample or load the 20 test sentence indices.
    """
    if INDICES_FILE.exists():
        with open(INDICES_FILE, "r", encoding="utf-8") as f:
            indices = json.load(f)
            logger.info(f"Loaded existing 20 structural check test indices from {INDICES_FILE}")
            return indices

    rng = random.Random(STRUCTURAL_SAMPLE_SEED)
    indices = sorted(rng.sample(range(total_test_sentences), STRUCTURAL_SAMPLE_SIZE))
    with open(INDICES_FILE, "w", encoding="utf-8") as f:
        json.dump(indices, f, indent=2)
    logger.info(f"Sampled and saved {len(indices)} test indices for structural evaluation to {INDICES_FILE}")
    return indices

def evaluate_structural_mst(
    embeddings_dict: Dict[str, np.ndarray],
    default_dim: int = 100
) -> Dict[str, Any]:
    """
    Perform bonus structural evaluation on the 20 sampled GUM test sentences.
    Computes MST over cosine distances and compares to gold dependency edges.
    """
    ds = load_ud_gum_dataset()
    test_split = ds["test"]
    indices = get_structural_check_indices(len(test_split))

    sentence_results = []
    total_correct = 0
    total_gold = 0

    for idx in indices:
        item = test_split[idx]
        tokens = [str(t).lower() for t in item["tokens"]]
        heads = item["head"]
        n_words = len(tokens)

        if n_words < 2:
            continue

        # Get vectors
        vecs = []
        for t in tokens:
            if t in embeddings_dict:
                vecs.append(embeddings_dict[t])
            else:
                # OOV: use zero vector
                vecs.append(np.zeros(default_dim, dtype=np.float32))

        # Compute pairwise cosine distance matrix
        dist_mat = np.zeros((n_words, n_words), dtype=np.float32)
        for i in range(n_words):
            for j in range(i + 1, n_words):
                sim = compute_cosine_similarity(vecs[i], vecs[j])
                dist = 1.0 - sim
                dist_mat[i, j] = dist
                dist_mat[j, i] = dist

        # MST
        pred_edges = compute_mst_undirected_edges(dist_mat)
        gold_edges = extract_gold_undirected_edges(heads)

        acc, correct, gold_count = compute_uuas(pred_edges, gold_edges)
        total_correct += correct
        total_gold += gold_count

        sentence_results.append({
            "sentence_idx": idx,
            "sentence_length": n_words,
            "tokens": tokens,
            "uuas": acc,
            "correct_edges": correct,
            "gold_edges": gold_count
        })

    overall_agreement = (total_correct / total_gold) if total_gold > 0 else 0.0
    macro_agreement = float(np.mean([r["uuas"] for r in sentence_results])) if sentence_results else 0.0

    results = {
        "overall_attachment_agreement": overall_agreement,
        "macro_attachment_agreement": macro_agreement,
        "total_correct_edges": total_correct,
        "total_gold_edges": total_gold,
        "num_sentences": len(sentence_results),
        "sampled_indices": indices,
        "sentence_breakdown": sentence_results
    }
    logger.info(f"Structural MST Check: Overall UUAS={overall_agreement:.4f} ({total_correct}/{total_gold} edges), Macro UUAS={macro_agreement:.4f}")
    return results

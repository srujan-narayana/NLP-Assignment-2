import logging
import numpy as np
from typing import Dict, List, Tuple, Any
from common.metrics import compute_cosine_similarity, compute_spearman_correlation, solve_analogy_3cosadd
from common.data_utils import load_wordsim353_dataset, load_google_analogies

logger = logging.getLogger(__name__)

def evaluate_wordsim353(
    embeddings_dict: Dict[str, np.ndarray]
) -> Dict[str, Any]:
    """
    Evaluate on WordSim-353 benchmark.
    Preserves original benchmark ordering.
    """
    pairs = load_wordsim353_dataset()
    human_scores = []
    model_scores = []
    evaluated_pairs = []
    oov_pairs = []

    for w1, w2, human_score in pairs:
        if w1 in embeddings_dict and w2 in embeddings_dict:
            sim = compute_cosine_similarity(embeddings_dict[w1], embeddings_dict[w2])
            human_scores.append(human_score)
            model_scores.append(sim)
            evaluated_pairs.append((w1, w2, human_score, sim))
        else:
            oov_pairs.append((w1, w2, human_score))

    spearman_corr, p_value = compute_spearman_correlation(human_scores, model_scores)
    
    results = {
        "spearman_correlation": spearman_corr,
        "p_value": p_value,
        "total_pairs": len(pairs),
        "evaluated_count": len(evaluated_pairs),
        "oov_count": len(oov_pairs),
        "coverage": len(evaluated_pairs) / len(pairs) if len(pairs) > 0 else 0.0
    }
    logger.info(f"WordSim-353: Spearman ρ={spearman_corr:.4f} (p={p_value:.2e}), Evaluated={len(evaluated_pairs)}/{len(pairs)} (OOV={len(oov_pairs)})")
    return results

def evaluate_google_analogy(
    embeddings_dict: Dict[str, np.ndarray],
    max_questions: int = None
) -> Dict[str, Any]:
    """
    Evaluate on Google Analogy questions using 3CosAdd.
    Drops questions containing OOV words and reports dropped count.
    """
    questions = load_google_analogies()
    if max_questions is not None:
        questions = questions[:max_questions]

    vocab = list(embeddings_dict.keys())
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    
    # Precompute normalized matrix for fast vector operations
    matrix = np.array([embeddings_dict[w] for w in vocab], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix_normed = matrix / norms

    evaluated = 0
    correct = 0
    dropped = 0
    by_category = {}

    for cat, a, b, c, d in questions:
        if cat not in by_category:
            by_category[cat] = {"correct": 0, "evaluated": 0, "dropped": 0}

        # Check OOV
        if a not in word_to_idx or b not in word_to_idx or c not in word_to_idx or d not in word_to_idx:
            dropped += 1
            by_category[cat]["dropped"] += 1
            continue

        evaluated += 1
        by_category[cat]["evaluated"] += 1

        pred = solve_analogy_3cosadd(
            embeddings_dict[a],
            embeddings_dict[b],
            embeddings_dict[c],
            vocab,
            matrix_normed,
            word_to_idx,
            exclude_words={a, b, c}
        )

        if pred == d:
            correct += 1
            by_category[cat]["correct"] += 1

    accuracy = (correct / evaluated) if evaluated > 0 else 0.0
    results = {
        "accuracy": accuracy,
        "correct": correct,
        "evaluated": evaluated,
        "dropped": dropped,
        "total_questions": len(questions),
        "by_category": by_category
    }
    logger.info(f"Google Analogy: Accuracy={accuracy:.4f} ({correct}/{evaluated}), Dropped OOV={dropped}/{len(questions)}")
    return results

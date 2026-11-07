from collections import defaultdict
from typing import Sequence

import numpy as np

from common.metrics import compute_cosine_similarity, compute_spearman_correlation
from common.metrics import solve_analogy_3cosadd


def evaluate_wordsim353(
    embeddings: dict[str, np.ndarray],
    pairs: Sequence[tuple[str, str, float]],
) -> dict[str, float | int]:
    gold: list[float] = []
    predicted: list[float] = []
    oov = 0
    for first, second, score in pairs:
        first, second = first.lower(), second.lower()
        if first not in embeddings or second not in embeddings:
            oov += 1
            continue
        gold.append(float(score))
        predicted.append(compute_cosine_similarity(embeddings[first], embeddings[second]))
    correlation, p_value = compute_spearman_correlation(gold, predicted)
    total = len(pairs)
    return {
        "spearman_correlation": correlation,
        "p_value": p_value,
        "total_pairs": total,
        "evaluated_count": len(gold),
        "oov_count": oov,
        "coverage": len(gold) / total if total else 0.0,
    }


def evaluate_google_analogy(
    embeddings: dict[str, np.ndarray],
    questions: Sequence[tuple[str, str, str, str, str]],
    batch_size: int = 128,
) -> dict[str, object]:
    """Evaluate normalized 3CosAdd in batches, excluding each question's inputs."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    vocabulary = list(embeddings)
    if not vocabulary:
        return {
            "accuracy": 0.0,
            "correct": 0,
            "evaluated": 0,
            "dropped": len(questions),
            "total_questions": len(questions),
            "by_category": {},
        }

    raw_matrix = np.asarray([embeddings[word] for word in vocabulary], dtype=np.float32)
    if raw_matrix.ndim != 2 or not np.isfinite(raw_matrix).all():
        raise ValueError("Embedding vectors must form a finite two-dimensional matrix.")
    norms = np.linalg.norm(raw_matrix, axis=1, keepdims=True)
    normalized = np.divide(
        raw_matrix,
        norms,
        out=np.zeros_like(raw_matrix),
        where=norms > 0.0,
    )
    word_to_idx = {word: index for index, word in enumerate(vocabulary)}

    eligible: list[tuple[str, int, int, int, str]] = []
    by_category: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "evaluated": 0, "dropped": 0}
    )
    dropped = 0
    for category, a, b, c, expected in questions:
        category = category or "uncategorized"
        a, b, c, expected = (word.lower() for word in (a, b, c, expected))
        indices = [word_to_idx.get(word) for word in (a, b, c, expected)]
        if any(index is None for index in indices):
            dropped += 1
            by_category[category]["dropped"] += 1
            continue
        ia, ib, ic, _ = indices
        eligible.append((category, int(ia), int(ib), int(ic), expected))
        by_category[category]["evaluated"] += 1

    correct = 0
    for start in range(0, len(eligible), batch_size):
        batch = eligible[start : start + batch_size]
        a_indices = np.fromiter((item[1] for item in batch), dtype=np.int64)
        b_indices = np.fromiter((item[2] for item in batch), dtype=np.int64)
        c_indices = np.fromiter((item[3] for item in batch), dtype=np.int64)
        targets = normalized[b_indices] - normalized[a_indices] + normalized[c_indices]
        target_norms = np.linalg.norm(targets, axis=1, keepdims=True)
        targets = np.divide(
            targets,
            target_norms,
            out=np.zeros_like(targets),
            where=target_norms > 0.0,
        )
        scores = targets @ normalized.T
        for row, item in enumerate(batch):
            category, ia, ib, ic, expected = item
            scores[row, [ia, ib, ic]] = -np.inf
            predicted = vocabulary[int(np.argmax(scores[row]))]
            if predicted == expected:
                correct += 1
                by_category[category]["correct"] += 1

    evaluated = len(eligible)
    return {
        "accuracy": correct / evaluated if evaluated else 0.0,
        "correct": correct,
        "evaluated": evaluated,
        "dropped": dropped,
        "total_questions": len(questions),
        "by_category": dict(by_category),
    }


def predict_analogy(
    a: str, b: str, c: str, embeddings: dict[str, np.ndarray]
) -> str | None:
    """Small public helper used by tests and downstream callers."""
    vocabulary = list(embeddings)
    if any(word not in embeddings for word in (a, b, c)):
        return None
    matrix = np.asarray([embeddings[word] for word in vocabulary], dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
    return solve_analogy_3cosadd(
        embeddings[a],
        embeddings[b],
        embeddings[c],
        vocabulary,
        normalized,
        {word: index for index, word in enumerate(vocabulary)},
        {a, b, c},
    )

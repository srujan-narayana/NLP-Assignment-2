from typing import Any

import numpy as np
from scipy.stats import spearmanr


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    first = np.asarray(vec1, dtype=np.float64)
    second = np.asarray(vec2, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError("Cosine similarity requires vectors with matching shapes.")
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(first, second) / denominator)


def compute_spearman_correlation(
    first: list[float] | np.ndarray, second: list[float] | np.ndarray
) -> tuple[float, float]:
    if len(first) != len(second):
        raise ValueError("Spearman inputs must have the same length.")
    if len(first) < 2:
        return 0.0, 1.0
    result = spearmanr(first, second)
    correlation = float(result.statistic)
    p_value = float(result.pvalue)
    if not np.isfinite(correlation):
        return 0.0, 1.0
    if not np.isfinite(p_value):
        p_value = 1.0
    return correlation, p_value


def solve_analogy_3cosadd(
    a_vec: np.ndarray,
    b_vec: np.ndarray,
    c_vec: np.ndarray,
    candidate_vocab: list[str],
    candidate_matrix_normed: np.ndarray,
    word_to_idx: dict[str, int],
    exclude_words: set[str],
) -> str | None:
    """Solve with normalized 3CosAdd vectors and exclude all three query words."""
    vectors = [np.asarray(vector, dtype=np.float64) for vector in (a_vec, b_vec, c_vec)]
    normalized: list[np.ndarray] = []
    for vector in vectors:
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return None
        normalized.append(vector / norm)
    target = normalized[1] - normalized[0] + normalized[2]
    target_norm = float(np.linalg.norm(target))
    if target_norm == 0.0:
        return None
    if len(candidate_vocab) != len(candidate_matrix_normed):
        raise ValueError("Candidate vocabulary and matrix row counts do not match.")
    similarities = np.asarray(candidate_matrix_normed) @ (target / target_norm)
    similarities = similarities.copy()
    for word in exclude_words:
        index = word_to_idx.get(word)
        if index is not None:
            similarities[index] = -np.inf
    if not np.isfinite(similarities).any():
        return None
    return candidate_vocab[int(np.argmax(similarities))]


def extract_gold_undirected_edges(heads: list[Any] | tuple[Any, ...]) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    n = len(heads)
    for dependent, raw_head in enumerate(heads):
        try:
            head = int(raw_head)
        except (TypeError, ValueError):
            continue
        if 1 <= head <= n and head - 1 != dependent:
            edges.add((min(dependent, head - 1), max(dependent, head - 1)))
    return edges


def compute_mst_undirected_edges(distance_matrix: np.ndarray) -> set[tuple[int, int]]:
    """Compute a deterministic complete-graph MST with Kruskal's algorithm."""
    distances = np.asarray(distance_matrix, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("MST input must be a square distance matrix.")
    if not np.isfinite(distances).all():
        raise ValueError("MST distances must be finite.")
    n = distances.shape[0]
    if n <= 1:
        return set()

    edges = [
        (float(distances[i, j]), i, j)
        for i in range(n)
        for j in range(i + 1, n)
    ]
    edges.sort()
    parent = list(range(n))
    size = [1] * n

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    selected: set[tuple[int, int]] = set()
    for _, left, right in edges:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            continue
        if size[root_left] < size[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        size[root_left] += size[root_right]
        selected.add((left, right))
        if len(selected) == n - 1:
            break
    return selected


def compute_uuas(
    predicted_edges: set[tuple[int, int]], gold_edges: set[tuple[int, int]]
) -> tuple[float, int, int]:
    if not gold_edges:
        return (1.0 if not predicted_edges else 0.0, 0, 0)
    correct = len(predicted_edges & gold_edges)
    return correct / len(gold_edges), correct, len(gold_edges)

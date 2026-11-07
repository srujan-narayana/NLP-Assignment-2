from collections import deque
from typing import Any

import numpy as np


def build_gold_structures(
    heads: list[Any] | tuple[Any, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate one dependency tree and return pairwise distances and root depths."""
    n = len(heads)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32), np.zeros(0, dtype=np.float32)
    normalized: list[int] = []
    for index, raw_head in enumerate(heads):
        try:
            head = int(raw_head)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Dependency head at token {index} is not an integer.") from exc
        if head < 0 or head > n:
            raise ValueError(f"Dependency head {head} at token {index} is outside 0..{n}.")
        if head == index + 1:
            raise ValueError(f"Token {index} cannot be its own dependency head.")
        normalized.append(head)
    roots = [index for index, head in enumerate(normalized) if head == 0]
    if len(roots) != 1:
        raise ValueError(f"A dependency tree must have exactly one root; found {len(roots)}.")

    adjacency: list[list[int]] = [[] for _ in range(n)]
    for dependent, head in enumerate(normalized):
        if head == 0:
            continue
        parent = head - 1
        adjacency[dependent].append(parent)
        adjacency[parent].append(dependent)

    distances = np.full((n, n), np.inf, dtype=np.float32)
    for start in range(n):
        distances[start, start] = 0.0
        queue: deque[int] = deque([start])
        while queue:
            node = queue.popleft()
            next_distance = distances[start, node] + 1.0
            for neighbor in adjacency[node]:
                if not np.isfinite(distances[start, neighbor]):
                    distances[start, neighbor] = next_distance
                    queue.append(neighbor)
    if not np.isfinite(distances).all():
        raise ValueError("Dependency heads do not form a connected tree.")

    depths = distances[roots[0]].copy()
    return distances, depths

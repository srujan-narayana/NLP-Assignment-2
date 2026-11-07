import logging
import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)

def build_gold_structures(heads: List[Any]) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Given head indices for a sentence (1-indexed, 0 = root):
    Returns:
        dist_matrix: (N, N) matrix of pairwise tree distances
        depth_array: (N,) array of tree depths from the root
        is_tree: bool indicating whether the dependency structure forms a valid connected tree
    """
    n = len(heads)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32), np.zeros(0, dtype=np.float32), True

    G = nx.Graph()
    for i in range(n):
        G.add_node(i)

    roots = []
    for i, h in enumerate(heads):
        try:
            h_val = int(h)
        except (ValueError, TypeError):
            h_val = 0

        if h_val == 0:
            roots.append(i)
        elif 1 <= h_val <= n and (h_val - 1) != i:
            parent = h_val - 1
            G.add_edge(i, parent)

    # All-pairs shortest path distance
    dist_matrix = np.zeros((n, n), dtype=np.float32)
    path_lengths = dict(nx.all_pairs_shortest_path_length(G))

    for i in range(n):
        for j in range(n):
            if j in path_lengths.get(i, {}):
                dist_matrix[i, j] = float(path_lengths[i][j])
            else:
                # Disconnected components fallback: assign large distance
                dist_matrix[i, j] = float(2 * n)

    # Compute depth from root
    depth_array = np.zeros(n, dtype=np.float32)
    primary_root = roots[0] if len(roots) > 0 else 0

    for i in range(n):
        # Find shortest distance to any root
        min_depth = None
        for r in (roots if len(roots) > 0 else [0]):
            if r in path_lengths.get(i, {}):
                d = path_lengths[i][r]
                if min_depth is None or d < min_depth:
                    min_depth = d
        if min_depth is not None:
            depth_array[i] = float(min_depth)
        else:
            depth_array[i] = float(n)

    is_tree = nx.is_tree(G) if n > 0 else True
    return dist_matrix, depth_array, is_tree

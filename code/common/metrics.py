import numpy as np
from scipy.stats import spearmanr
from typing import List, Tuple, Set, Dict, Any, Optional
import networkx as nx

def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))

def compute_spearman_correlation(scores1: List[float], scores2: List[float]) -> Tuple[float, float]:
    """Compute Spearman rank correlation and p-value between two score lists."""
    if len(scores1) < 2 or len(scores2) < 2:
        return 0.0, 1.0
    corr, pval = spearmanr(scores1, scores2)
    if np.isnan(corr):
        corr = 0.0
    return float(corr), float(pval)

def solve_analogy_3cosadd(
    a_vec: np.ndarray,
    b_vec: np.ndarray,
    c_vec: np.ndarray,
    candidate_vocab: List[str],
    candidate_matrix_normed: np.ndarray,
    word_to_idx: Dict[str, int],
    exclude_words: Set[str]
) -> Optional[str]:
    """
    Solve a : b :: c : ? using 3CosAdd: target = b - a + c.
    Find candidate in candidate_vocab maximizing cosine similarity,
    excluding words in exclude_words.
    """
    target = b_vec - a_vec + c_vec
    norm = np.linalg.norm(target)
    if norm == 0:
        return None
    target_normed = target / norm

    # Similarities with all vocabulary entries
    sims = np.dot(candidate_matrix_normed, target_normed)

    # Exclude query words by setting similarity to -inf
    for w in exclude_words:
        if w in word_to_idx:
            sims[word_to_idx[w]] = -np.inf

    best_idx = int(np.argmax(sims))
    if sims[best_idx] == -np.inf:
        return None
    return candidate_vocab[best_idx]

def extract_gold_undirected_edges(heads: List[Any], exclude_punctuation: bool = False, tokens: Optional[List[str]] = None) -> Set[Tuple[int, int]]:
    """
    Extract gold undirected dependency edges from CoNLL-U heads (1-indexed).
    Root (head == 0 or '0') has no sentence-internal head.
    Returns set of undirected edges (min(i, j), max(i, j)) with 0-indexed word indices.
    """
    gold_edges = set()
    n = len(heads)
    for i, h in enumerate(heads):
        try:
            h_val = int(h)
        except (ValueError, TypeError):
            continue
        if h_val > 0:
            parent = h_val - 1
            if parent < n and parent != i:
                edge = (min(i, parent), max(i, parent))
                gold_edges.add(edge)
    return gold_edges

def compute_mst_undirected_edges(distance_matrix: np.ndarray) -> Set[Tuple[int, int]]:
    """
    Given an n x n symmetric pairwise distance matrix,
    compute the Minimum Spanning Tree and return the set of undirected edges.
    """
    n = distance_matrix.shape[0]
    if n <= 1:
        return set()

    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
        for j in range(i + 1, n):
            G.add_edge(i, j, weight=float(distance_matrix[i, j]))

    mst = nx.minimum_spanning_tree(G)
    edges = set()
    for u, v in mst.edges():
        edges.add((min(u, v), max(u, v)))
    return edges

def compute_uuas(predicted_edges: Set[Tuple[int, int]], gold_edges: Set[Tuple[int, int]]) -> Tuple[float, int, int]:
    """
    Compute Undirected Unlabeled Attachment Score (UUAS).
    Returns (accuracy, correct_edges, total_gold_edges).
    """
    if len(gold_edges) == 0:
        return (1.0 if len(predicted_edges) == 0 else 0.0, 0, 0)
    correct = len(predicted_edges.intersection(gold_edges))
    total = len(gold_edges)
    return (correct / total, correct, total)

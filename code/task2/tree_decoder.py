import logging
import numpy as np
import torch
from typing import Set, Tuple, List, Dict, Any
from common.metrics import compute_mst_undirected_edges, extract_gold_undirected_edges, compute_uuas

logger = logging.getLogger(__name__)

def decode_tree_and_evaluate_uuas(
    pred_distance_matrix: np.ndarray,
    gold_heads: List[Any]
) -> Tuple[float, int, int]:
    """
    Decodes MST from distance matrix and compares to gold dependency edges.
    Returns:
        (uuas, correct_edges, total_gold_edges)
    """
    pred_edges = compute_mst_undirected_edges(pred_distance_matrix)
    gold_edges = extract_gold_undirected_edges(gold_heads)
    return compute_uuas(pred_edges, gold_edges)

import logging
import torch
import numpy as np
from typing import List, Tuple, Dict, Any
from common.metrics import compute_spearman_correlation
from task2.tree_decoder import decode_tree_and_evaluate_uuas

logger = logging.getLogger(__name__)

def evaluate_distance_probe_uuas(
    probe: torch.nn.Module,
    representations: List[torch.Tensor],
    gold_heads: List[List[Any]],
    device: torch.device
) -> Tuple[float, float, int, int]:
    """
    Evaluate distance probe by decoding MST trees on all sentences.
    Returns:
        overall_uuas: total_correct / total_gold
        macro_uuas: average per-sentence uuas
        total_correct: sum of correctly predicted edges
        total_gold: sum of gold edges
    """
    probe.eval()
    total_correct = 0
    total_gold = 0
    sentence_uuas = []

    with torch.no_grad():
        for rep, heads in zip(representations, gold_heads):
            n = rep.shape[0]
            if n <= 1:
                continue
            rep = rep.to(device)
            dist_sq = probe(rep).cpu().numpy()
            uuas, correct, gold_count = decode_tree_and_evaluate_uuas(dist_sq, heads)

            total_correct += correct
            total_gold += gold_count
            sentence_uuas.append(uuas)

    overall_uuas = (total_correct / total_gold) if total_gold > 0 else 0.0
    macro_uuas = float(np.mean(sentence_uuas)) if sentence_uuas else 0.0
    return overall_uuas, macro_uuas, total_correct, total_gold

def evaluate_depth_probe_correlation(
    probe: torch.nn.Module,
    representations: List[torch.Tensor],
    gold_depths: List[torch.Tensor],
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluate depth probe by computing Spearman correlation between predicted and gold depths.
    Returns:
        (mean_spearman_rho, p_value)
    """
    probe.eval()
    sentence_corrs = []

    with torch.no_grad():
        for rep, g_depth in zip(representations, gold_depths):
            n = rep.shape[0]
            if n <= 1:
                continue
            rep = rep.to(device)
            pred_depth = probe(rep).cpu().numpy()
            gold_d = g_depth.numpy()

            # Ignore if all depths are equal (e.g. 1-word or flat)
            if np.std(pred_depth) > 1e-6 and np.std(gold_d) > 1e-6:
                corr, _ = compute_spearman_correlation(list(pred_depth), list(gold_d))
                sentence_corrs.append(corr)

    mean_corr = float(np.mean(sentence_corrs)) if sentence_corrs else 0.0
    return mean_corr, 0.0

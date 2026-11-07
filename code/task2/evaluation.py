from typing import Sequence

import numpy as np
import torch

from common.data_utils import UDSentence
from common.metrics import (
    compute_mst_undirected_edges,
    compute_spearman_correlation,
    compute_uuas,
    extract_gold_undirected_edges,
)
from task2.distance_probe import StructuralDistanceProbe
from task2.depth_probe import StructuralDepthProbe


def evaluate_distance_probe_uuas(
    probe: StructuralDistanceProbe,
    representations: Sequence[torch.Tensor],
    sentences: Sequence[UDSentence],
    device: str | torch.device = "cpu",
) -> dict[str, float | int]:
    if len(representations) != len(sentences):
        raise ValueError("Representations and gold sentences must align.")
    probe.eval()
    total_correct = 0
    total_gold = 0
    sentence_scores: list[float] = []
    with torch.no_grad():
        for representation, sentence in zip(representations, sentences):
            if len(sentence.tokens) < 2:
                continue
            predicted_distance = probe(representation.to(device)).cpu().numpy()
            predicted_edges = compute_mst_undirected_edges(predicted_distance)
            gold_edges = extract_gold_undirected_edges(sentence.heads)
            score, correct, total = compute_uuas(predicted_edges, gold_edges)
            total_correct += correct
            total_gold += total
            sentence_scores.append(score)
    return {
        "uuas": total_correct / total_gold if total_gold else 0.0,
        "macro_uuas": float(np.mean(sentence_scores)) if sentence_scores else 0.0,
        "correct_edges": total_correct,
        "gold_edges": total_gold,
    }


def evaluate_depth_probe_correlation(
    probe: StructuralDepthProbe,
    representations: Sequence[torch.Tensor],
    gold_depths: Sequence[np.ndarray | torch.Tensor],
    device: str | torch.device = "cpu",
) -> dict[str, float | int]:
    """Report pooled token-level Spearman and macro sentence-level Spearman."""
    if len(representations) != len(gold_depths):
        raise ValueError("Representations and gold depths must align.")
    probe.eval()
    all_predicted: list[float] = []
    all_gold: list[float] = []
    sentence_correlations: list[float] = []
    with torch.no_grad():
        for representation, target in zip(representations, gold_depths):
            if representation.shape[0] <= 1:
                continue
            predicted = probe(representation.to(device)).cpu().numpy()
            gold = target.detach().cpu().numpy() if isinstance(target, torch.Tensor) else np.asarray(target)
            all_predicted.extend(float(value) for value in predicted)
            all_gold.extend(float(value) for value in gold)
            if np.std(predicted) > 1e-8 and np.std(gold) > 1e-8:
                correlation, _ = compute_spearman_correlation(predicted, gold)
                sentence_correlations.append(correlation)
    pooled, p_value = compute_spearman_correlation(all_predicted, all_gold)
    return {
        "spearman_correlation": pooled,
        "p_value": p_value,
        "macro_spearman": float(np.mean(sentence_correlations)) if sentence_correlations else 0.0,
        "evaluated_tokens": len(all_gold),
        "evaluated_sentences": len(sentence_correlations),
    }

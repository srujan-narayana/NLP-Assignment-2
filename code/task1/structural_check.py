import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np

from common.data_utils import UDSentence
from common.metrics import (
    compute_cosine_similarity,
    compute_mst_undirected_edges,
    compute_uuas,
    extract_gold_undirected_edges,
)


def get_structural_check_indices(
    total_sentences: int,
    path: str | Path,
    sample_size: int = 20,
    seed: int = 42,
) -> list[int]:
    if total_sentences < 1:
        raise ValueError("The test split must contain at least one sentence.")
    sample_count = min(sample_size, total_sentences)
    target = Path(path)
    if target.exists():
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("test_size") != total_sentences:
            raise ValueError(
                f"Saved structural sample {target} does not match this test split."
            )
        indices = payload.get("indices")
        if (
            not isinstance(indices, list)
            or len(indices) != sample_count
            or len(set(indices)) != len(indices)
            or any(not isinstance(index, int) or index < 0 or index >= total_sentences for index in indices)
        ):
            raise ValueError(f"Saved structural sample {target} is invalid.")
        return indices

    indices = sorted(random.Random(seed).sample(range(total_sentences), sample_count))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "seed": seed,
                "test_size": total_sentences,
                "requested_size": sample_size,
                "indices": indices,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return indices


def evaluate_structural_mst(
    embeddings: dict[str, np.ndarray],
    test_sentences: Sequence[UDSentence],
    sample_path: str | Path,
    sample_size: int = 20,
    seed: int = 42,
) -> dict[str, object]:
    indices = get_structural_check_indices(
        len(test_sentences), sample_path, sample_size=sample_size, seed=seed
    )
    dimension = next(iter(embeddings.values())).shape[0] if embeddings else 0
    total_correct = 0
    total_gold = 0
    oov_tokens = 0
    sentence_scores: list[float] = []
    details: list[dict[str, object]] = []

    for index in indices:
        sentence = test_sentences[index]
        if len(sentence.tokens) < 2:
            details.append(
                {"sentence_idx": index, "sentence_length": len(sentence.tokens), "uuas": 1.0}
            )
            continue
        vectors: list[np.ndarray] = []
        for token in sentence.tokens:
            word = token.lower()
            if word in embeddings:
                vectors.append(np.asarray(embeddings[word], dtype=np.float32))
            else:
                vectors.append(np.zeros(dimension, dtype=np.float32))
                oov_tokens += 1

        distances = np.zeros((len(vectors), len(vectors)), dtype=np.float64)
        for left in range(len(vectors)):
            for right in range(left + 1, len(vectors)):
                distance = 1.0 - compute_cosine_similarity(vectors[left], vectors[right])
                distances[left, right] = distances[right, left] = distance

        predicted = compute_mst_undirected_edges(distances)
        gold = extract_gold_undirected_edges(sentence.heads)
        score, correct, gold_count = compute_uuas(predicted, gold)
        sentence_scores.append(score)
        total_correct += correct
        total_gold += gold_count
        details.append(
            {
                "sentence_idx": index,
                "sentence_length": len(sentence.tokens),
                "uuas": score,
                "correct_edges": correct,
                "gold_edges": gold_count,
            }
        )

    return {
        "overall_attachment_agreement": total_correct / total_gold if total_gold else 0.0,
        "macro_attachment_agreement": float(np.mean(sentence_scores)) if sentence_scores else 0.0,
        "total_correct_edges": total_correct,
        "total_gold_edges": total_gold,
        "num_sentences": len(indices),
        "oov_tokens": oov_tokens,
        "sampled_indices": indices,
        "sentence_breakdown": details,
    }

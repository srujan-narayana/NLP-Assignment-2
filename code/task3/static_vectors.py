from collections import Counter
from typing import Sequence

import numpy as np
import torch

from common.data_utils import UDSentence


def construct_static_embeddings_from_bert(
    sentences: Sequence[UDSentence],
    word_representations: Sequence[torch.Tensor],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Average token vectors online, using memory proportional to vocabulary size."""
    if len(sentences) != len(word_representations):
        raise ValueError("Sentence and representation counts do not match.")
    if not word_representations:
        return {}, {}
    dimension = int(word_representations[0].shape[-1])
    sums: dict[str, np.ndarray] = {}
    frequencies: Counter[str] = Counter()

    for sentence, representation in zip(sentences, word_representations):
        if representation.ndim != 2 or representation.shape != (
            len(sentence.tokens),
            dimension,
        ):
            raise ValueError("Every sentence representation must align with its tokens.")
        values = representation.detach().cpu().numpy().astype(np.float64, copy=False)
        for token, vector in zip(sentence.tokens, values):
            word = token.lower()
            if word not in sums:
                sums[word] = np.zeros(dimension, dtype=np.float64)
            sums[word] += vector
            frequencies[word] += 1

    embeddings = {
        word: (total / frequencies[word]).astype(np.float32)
        for word, total in sums.items()
    }
    return embeddings, dict(frequencies)

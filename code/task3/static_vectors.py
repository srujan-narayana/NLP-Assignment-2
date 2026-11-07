import logging
from collections import defaultdict
import numpy as np
import torch
from typing import List, Dict, Tuple, Any

logger = logging.getLogger(__name__)

def construct_static_embeddings_from_bert(
    sentences_tokens: List[List[str]],
    word_representations: List[torch.Tensor]
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """
    Construct static word vectors by averaging contextual embeddings across all occurrences
    in the training corpus.
    
    Returns:
        static_embeddings: dict mapping lowercase word -> 768-dim numpy array
        word_frequencies: dict mapping lowercase word -> occurrence count
    """
    accumulators = defaultdict(list)
    word_frequencies = defaultdict(int)

    logger.info("Aggregating contextual token representations by word type...")
    for tokens, reps in zip(sentences_tokens, word_representations):
        n = min(len(tokens), reps.shape[0])
        for i in range(n):
            word = tokens[i].lower()
            vec = reps[i].numpy()
            accumulators[word].append(vec)
            word_frequencies[word] += 1

    static_embeddings = {}
    for word, vecs in accumulators.items():
        # Average contextual vectors
        mean_vec = np.mean(vecs, axis=0)
        static_embeddings[word] = mean_vec.astype(np.float32)

    logger.info(f"Built BERT static vector vocabulary: {len(static_embeddings)} unique words.")
    return static_embeddings, dict(word_frequencies)

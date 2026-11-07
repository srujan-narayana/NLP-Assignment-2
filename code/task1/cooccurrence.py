import logging
from collections import defaultdict, Counter
import numpy as np
import scipy.sparse as sp
from typing import List, Dict, Tuple
from config import WINDOW_SIZE

logger = logging.getLogger(__name__)

def build_cooccurrence_matrix(
    sentences: List[List[str]],
    word_to_idx: Dict[str, int],
    window_size: int = WINDOW_SIZE
) -> sp.csr_matrix:
    """
    Construct a symmetric word-word co-occurrence matrix over the sentences.
    Co-occurrences never cross sentence boundaries.
    """
    vocab_size = len(word_to_idx)
    logger.info(f"Building co-occurrence matrix (vocab_size={vocab_size}, window_size={window_size})...")

    pair_counts = defaultdict(int)

    for sent in sentences:
        indices = [word_to_idx[w] for w in sent if w in word_to_idx]
        sent_len = len(indices)
        for i, target_idx in enumerate(indices):
            start = max(0, i - window_size)
            end = min(sent_len, i + window_size + 1)
            for j in range(start, end):
                if i != j:
                    context_idx = indices[j]
                    pair_counts[(target_idx, context_idx)] += 1

    rows, cols, data = [], [], []
    for (i, j), count in pair_counts.items():
        rows.append(i)
        cols.append(j)
        data.append(count)

    cooc_matrix = sp.csr_matrix((data, (rows, cols)), shape=(vocab_size, vocab_size), dtype=np.float64)
    logger.info(f"Co-occurrence matrix built with {cooc_matrix.nnz} non-zero entries.")
    return cooc_matrix

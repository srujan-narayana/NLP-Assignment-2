import numpy as np
import scipy.sparse as sp


def build_cooccurrence_matrix(
    sentences: list[list[str] | tuple[str, ...]],
    word_to_idx: dict[str, int],
    window_size: int = 5,
) -> sp.csr_matrix:
    """Count symmetric, within-sentence co-occurrences without compressing OOV gaps."""
    if window_size < 1:
        raise ValueError("window_size must be at least 1.")
    vocab_size = len(word_to_idx)
    if vocab_size == 0:
        return sp.csr_matrix((0, 0), dtype=np.float64)

    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    for sentence in sentences:
        # -1 marks tokens outside the chosen vocabulary. Keep their original positions
        # so filtering cannot pull words from beyond the specified context window.
        ids = np.fromiter(
            (word_to_idx.get(str(token).lower(), -1) for token in sentence),
            dtype=np.int64,
        )
        for distance in range(1, min(window_size, max(0, len(ids) - 1)) + 1):
            left = ids[:-distance]
            right = ids[distance:]
            keep = (left >= 0) & (right >= 0)
            if not np.any(keep):
                continue
            left_kept = left[keep]
            right_kept = right[keep]
            row_parts.extend((left_kept, right_kept))
            col_parts.extend((right_kept, left_kept))

    if not row_parts:
        return sp.csr_matrix((vocab_size, vocab_size), dtype=np.float64)
    rows = np.concatenate(row_parts)
    cols = np.concatenate(col_parts)
    data = np.ones(rows.shape[0], dtype=np.float64)
    matrix = sp.coo_matrix((data, (rows, cols)), shape=(vocab_size, vocab_size))
    matrix.sum_duplicates()
    return matrix.tocsr()

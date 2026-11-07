import logging
import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)

def compute_ppmi_matrix(cooc_matrix: sp.csr_matrix) -> sp.csr_matrix:
    """
    Calculate the Positive Pointwise Mutual Information (PPMI) matrix
    efficiently using sparse operations.
    
    PMI(w, c) = log2( (C(w, c) * N) / (C(w) * C(c)) )
    PPMI(w, c) = max(0, PMI(w, c))
    """
    logger.info("Computing PPMI matrix...")
    cooc_coo = cooc_matrix.tocoo()
    
    # Total sum of co-occurrences
    total_sum = float(cooc_coo.data.sum())
    if total_sum == 0:
        logger.warning("Total co-occurrences sum is 0.")
        return cooc_matrix.copy()

    # Marginal sums
    row_sums = np.array(cooc_matrix.sum(axis=1)).flatten()  # C(w)
    col_sums = np.array(cooc_matrix.sum(axis=0)).flatten()  # C(c)

    rows = cooc_coo.row
    cols = cooc_coo.col
    data = cooc_coo.data

    # Avoid divide-by-zero for any zero marginals
    expected = (row_sums[rows] * col_sums[cols]) / total_sum
    # Avoid zero division
    valid = (expected > 0) & (data > 0)
    
    pmi_data = np.zeros_like(data, dtype=np.float64)
    pmi_data[valid] = np.log2(data[valid] / expected[valid])
    
    # PPMI: clamp at 0
    ppmi_data = np.maximum(0.0, pmi_data)
    
    # Filter out zeros to keep it truly sparse
    positive_mask = ppmi_data > 0
    new_rows = rows[positive_mask]
    new_cols = cols[positive_mask]
    new_data = ppmi_data[positive_mask]

    ppmi_matrix = sp.csr_matrix(
        (new_data, (new_rows, new_cols)),
        shape=cooc_matrix.shape,
        dtype=np.float64
    )
    logger.info(f"PPMI computed: {ppmi_matrix.nnz} positive entries remaining.")
    return ppmi_matrix

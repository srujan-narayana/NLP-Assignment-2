import numpy as np
import scipy.sparse as sp


def compute_ppmi_matrix(cooccurrence: sp.spmatrix) -> sp.csr_matrix:
    matrix = cooccurrence.tocsr().astype(np.float64)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("The co-occurrence matrix must be square.")
    if np.any(matrix.data < 0) or not np.isfinite(matrix.data).all():
        raise ValueError("Co-occurrence counts must be finite and non-negative.")
    total = float(matrix.sum())
    if total <= 0.0 or matrix.nnz == 0:
        return sp.csr_matrix(matrix.shape, dtype=np.float64)

    coo = matrix.tocoo()
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    col_sums = np.asarray(matrix.sum(axis=0)).ravel()
    numerator = coo.data * total
    denominator = row_sums[coo.row] * col_sums[coo.col]
    valid = (numerator > 0.0) & (denominator > 0.0)
    values = np.zeros_like(coo.data)
    values[valid] = np.maximum(
        0.0, np.log2(numerator[valid] / denominator[valid])
    )
    positive = values > 0.0
    result = sp.coo_matrix(
        (values[positive], (coo.row[positive], coo.col[positive])),
        shape=matrix.shape,
        dtype=np.float64,
    )
    result.sum_duplicates()
    return result.tocsr()

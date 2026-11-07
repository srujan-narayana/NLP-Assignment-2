import logging
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD
from typing import Dict, List, Tuple
from config import EMBEDDING_DIM, RANDOM_SEED

logger = logging.getLogger(__name__)

def compute_svd_embeddings(
    ppmi_matrix: sp.csr_matrix,
    dim: int = EMBEDDING_DIM,
    random_state: int = RANDOM_SEED
) -> np.ndarray:
    """
    Perform Truncated SVD on PPMI matrix to obtain dense word vectors.
    Returns:
        embeddings: (vocab_size, dim) matrix of embeddings.
    """
    vocab_size = ppmi_matrix.shape[0]
    target_dim = min(dim, vocab_size - 1)
    logger.info(f"Computing Truncated SVD (dim={target_dim}, vocab_size={vocab_size})...")

    # Using sklearn TruncatedSVD for deterministic reproducibility and robustness
    tsvd = TruncatedSVD(n_components=target_dim, algorithm="randomized", random_state=random_state)
    embeddings = tsvd.fit_transform(ppmi_matrix)
    
    # If vocab_size was smaller than dim, pad with zeros to exact dim
    if target_dim < dim:
        pad = np.zeros((vocab_size, dim - target_dim))
        embeddings = np.hstack([embeddings, pad])

    logger.info(f"SVD embeddings shape: {embeddings.shape}, explained variance ratio sum: {tsvd.explained_variance_ratio_.sum():.4f}")
    return embeddings

def build_embedding_dict(
    embeddings: np.ndarray,
    idx_to_word: List[str]
) -> Dict[str, np.ndarray]:
    """Map words to their 1D numpy vector embeddings."""
    return {word: embeddings[i] for i, word in enumerate(idx_to_word)}

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD


def compute_svd_embeddings(
    ppmi: sp.spmatrix, dim: int = 100, random_state: int = 42
) -> np.ndarray:
    if dim < 1:
        raise ValueError("Embedding dimension must be positive.")
    matrix = ppmi.tocsr()
    vocab_size = matrix.shape[0]
    embeddings = np.zeros((vocab_size, dim), dtype=np.float32)
    rank = min(dim, vocab_size - 1)
    if rank <= 0 or matrix.nnz == 0:
        return embeddings

    model = TruncatedSVD(
        n_components=rank, algorithm="randomized", random_state=random_state
    )
    reduced = model.fit_transform(matrix).astype(np.float32, copy=False)
    embeddings[:, :rank] = reduced
    return embeddings


def build_embedding_dict(
    embeddings: np.ndarray, vocabulary: list[str]
) -> dict[str, np.ndarray]:
    if embeddings.ndim != 2 or embeddings.shape[0] != len(vocabulary):
        raise ValueError("Embedding rows must align with the vocabulary.")
    return {word: embeddings[index] for index, word in enumerate(vocabulary)}

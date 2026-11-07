import logging
from typing import Dict, Any
import numpy as np
from task1.structural_check import evaluate_structural_mst

logger = logging.getLogger(__name__)

def evaluate_bert_static_structural(
    static_embeddings: Dict[str, np.ndarray],
    default_dim: int = 768
) -> Dict[str, Any]:
    """
    Run the exact same 20-sentence structural check using BERT-derived static vectors.
    """
    logger.info("Evaluating structural MST agreement on BERT-derived static vectors...")
    return evaluate_structural_mst(static_embeddings, default_dim=default_dim)

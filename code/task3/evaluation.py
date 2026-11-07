import logging
from typing import Dict, Any
import numpy as np
from task1.evaluation import evaluate_wordsim353, evaluate_google_analogy

logger = logging.getLogger(__name__)

def evaluate_bert_static_embeddings(
    static_embeddings: Dict[str, np.ndarray]
) -> Dict[str, Any]:
    """
    Run WordSim-353 and Google Analogy evaluations on BERT static embeddings.
    """
    logger.info("Evaluating WordSim-353 on BERT-derived static embeddings...")
    wordsim_res = evaluate_wordsim353(static_embeddings)

    logger.info("Evaluating Google Analogy on BERT-derived static embeddings...")
    analogy_res = evaluate_google_analogy(static_embeddings)

    return {
        "wordsim353": wordsim_res,
        "google_analogy": analogy_res
    }

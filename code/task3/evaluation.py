from typing import Sequence

import numpy as np

from task1.evaluation import evaluate_google_analogy, evaluate_wordsim353


def evaluate_bert_static_embeddings(
    embeddings: dict[str, np.ndarray],
    wordsim_pairs: Sequence[tuple[str, str, float]],
    analogy_questions: Sequence[tuple[str, str, str, str, str]],
) -> dict[str, object]:
    return {
        "wordsim353": evaluate_wordsim353(embeddings, wordsim_pairs),
        "google_analogy": evaluate_google_analogy(embeddings, analogy_questions),
    }

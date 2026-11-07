import logging
import torch
from typing import List, Dict, Any
from task2.bert_extractor import BertLayerExtractor
from config import BERT_MODEL_NAME

logger = logging.getLogger(__name__)

def extract_layer_representations(
    sentences_tokens: List[List[str]],
    best_layer: int,
    device: torch.device = None
) -> List[torch.Tensor]:
    """
    Extract word representations for the selected layer l*.
    Uses disk cache if available.
    """
    extractor = BertLayerExtractor(model_name=BERT_MODEL_NAME, device=str(device) if device else None)
    cached_reps = extractor.get_or_extract_representations("train", sentences_tokens, [best_layer])
    return cached_reps[best_layer]

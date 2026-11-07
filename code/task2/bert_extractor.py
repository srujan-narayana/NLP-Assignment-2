import logging
import torch
from transformers import BertTokenizer, BertModel
from typing import List, Dict, Tuple, Any
from pathlib import Path

from config import BERT_MODEL_NAME, DATA_DIR, MAX_SEQ_LEN
from task2.alignment import align_words_to_subwords, pool_word_representations

logger = logging.getLogger(__name__)

CACHE_DIR = DATA_DIR / "bert_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

class BertLayerExtractor:
    def __init__(self, model_name: str = BERT_MODEL_NAME, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Initializing BertLayerExtractor ({model_name}) on device: {self.device}")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name, output_hidden_states=True)
        self.model.to(self.device)
        self.model.eval()

        # Freeze transformer weights completely
        for param in self.model.parameters():
            param.requires_grad = False

    def extract_sentence_representations(
        self,
        sentences_tokens: List[List[str]],
        layer_indices: List[int] = list(range(1, 13))
    ) -> Dict[int, List[torch.Tensor]]:
        """
        Extract word-level representations for specified layers across sentences.
        Returns:
            layer_dict: dict mapping layer_num (1..12) -> list of (N_words, 768) torch.FloatTensor
        """
        layer_dict = {l: [] for l in layer_indices}

        with torch.no_grad():
            for s_idx, words in enumerate(sentences_tokens):
                if len(words) == 0:
                    for l in layer_indices:
                        layer_dict[l].append(torch.empty((0, 768), dtype=torch.float32))
                    continue

                input_ids, word_subword_map = align_words_to_subwords(words, self.tokenizer)
                
                # Truncate if exceeding max length
                if len(input_ids) > MAX_SEQ_LEN:
                    input_ids = input_ids[:MAX_SEQ_LEN - 1] + [self.tokenizer.sep_token_id]
                    # Filter word map to only keep indices within bounds
                    filtered_map = []
                    for idxs in word_subword_map:
                        valid_idxs = [idx for idx in idxs if idx < MAX_SEQ_LEN - 1]
                        if valid_idxs:
                            filtered_map.append(valid_idxs)
                        else:
                            filtered_map.append([MAX_SEQ_LEN - 2])
                    word_subword_map = filtered_map

                input_tensor = torch.tensor([input_ids], device=self.device)
                outputs = self.model(input_tensor)
                # outputs.hidden_states is tuple of (embedding_output, layer1, ..., layer12)
                # index 0 is embedding, index 1 is layer 1, ..., index 12 is layer 12
                all_hidden_states = outputs.hidden_states

                for l in layer_indices:
                    layer_hidden = all_hidden_states[l][0]  # (seq_len, 768)
                    word_reps = pool_word_representations(layer_hidden, word_subword_map, strategy="mean")
                    # Store on CPU to avoid filling GPU RAM
                    layer_dict[l].append(word_reps.cpu())

                if (s_idx + 1) % 500 == 0 or (s_idx + 1) == len(sentences_tokens):
                    logger.info(f"Extracted representations for {s_idx + 1}/{len(sentences_tokens)} sentences.")

        return layer_dict

    def get_or_extract_representations(
        self,
        split_name: str,
        sentences_tokens: List[List[str]],
        layer_indices: List[int] = list(range(1, 13))
    ) -> Dict[int, List[torch.Tensor]]:
        """
        Check disk cache first; if not present, extract and cache.
        """
        cache_path = CACHE_DIR / f"bert_{split_name}_layers.pt"
        if cache_path.exists():
            logger.info(f"Loading cached representations from {cache_path}...")
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            # Verify all requested layers are present
            if all(l in cached for l in layer_indices):
                return cached

        logger.info(f"Cache not found or incomplete. Extracting representations for '{split_name}'...")
        reps = self.extract_sentence_representations(sentences_tokens, layer_indices)
        torch.save(reps, cache_path)
        logger.info(f"Saved extracted representations to {cache_path}")
        return reps

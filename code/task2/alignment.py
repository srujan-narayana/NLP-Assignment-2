import logging
from typing import List, Dict, Tuple, Any
import torch

logger = logging.getLogger(__name__)

def align_words_to_subwords(
    words: List[str],
    tokenizer: Any
) -> Tuple[List[int], List[List[int]]]:
    """
    Given a list of words from a sentence:
    Returns:
        input_ids: full sequence token IDs starting with [CLS] and ending with [SEP]
        word_to_subword_indices: list of subword index lists for each original word
    """
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    input_ids = [cls_id]
    word_to_subword_indices = []

    for word in words:
        # Tokenize word without special tokens
        subwords = tokenizer.encode(word, add_special_tokens=False)
        if len(subwords) == 0:
            # Fallback for empty/whitespace string
            subwords = [tokenizer.unk_token_id]

        start_idx = len(input_ids)
        input_ids.extend(subwords)
        end_idx = len(input_ids)
        word_to_subword_indices.append(list(range(start_idx, end_idx)))

    input_ids.append(sep_id)
    return input_ids, word_to_subword_indices

def pool_word_representations(
    sentence_hidden_states: torch.Tensor,
    word_to_subword_indices: List[List[int]],
    strategy: str = "mean"
) -> torch.Tensor:
    """
    Pool subword vectors into a single vector per original word.
    Args:
        sentence_hidden_states: (seq_len, hidden_dim) tensor
        word_to_subword_indices: list of index lists
        strategy: 'mean' or 'first'
    Returns:
        word_representations: (num_words, hidden_dim) tensor
    """
    word_reps = []
    for subword_indices in word_to_subword_indices:
        # subword_indices is list of ints
        sub_vecs = sentence_hidden_states[subword_indices]  # (k, hidden_dim)
        if strategy == "first":
            rep = sub_vecs[0]
        else:
            rep = sub_vecs.mean(dim=0)
        word_reps.append(rep)

    if len(word_reps) == 0:
        return torch.empty((0, sentence_hidden_states.shape[-1]), device=sentence_hidden_states.device)
    return torch.stack(word_reps, dim=0)

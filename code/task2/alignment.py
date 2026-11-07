from typing import Any

import torch


def align_words_to_subwords(
    words: list[str], tokenizer: Any
) -> tuple[list[int], list[list[int]]]:
    """Return BERT input IDs and the subword positions for every original word."""
    if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
        raise ValueError("The selected tokenizer must define CLS and SEP token IDs.")
    input_ids = [int(tokenizer.cls_token_id)]
    word_indices: list[list[int]] = []
    for word in words:
        pieces = tokenizer.encode(str(word), add_special_tokens=False)
        if not pieces:
            if tokenizer.unk_token_id is None:
                raise ValueError(f"Tokenizer could not encode word {word!r}.")
            pieces = [int(tokenizer.unk_token_id)]
        start = len(input_ids)
        input_ids.extend(int(piece) for piece in pieces)
        word_indices.append(list(range(start, len(input_ids))))
    input_ids.append(int(tokenizer.sep_token_id))
    return input_ids, word_indices


def pool_word_representations(
    hidden_states: torch.Tensor,
    word_to_subword_indices: list[list[int]],
    strategy: str = "mean",
) -> torch.Tensor:
    if strategy not in {"mean", "first"}:
        raise ValueError("Pooling strategy must be 'mean' or 'first'.")
    pooled: list[torch.Tensor] = []
    for indices in word_to_subword_indices:
        if not indices:
            raise ValueError("Every word must align to at least one subword.")
        pieces = hidden_states[indices]
        pooled.append(pieces[0] if strategy == "first" else pieces.mean(dim=0))
    if not pooled:
        return hidden_states.new_empty((0, hidden_states.shape[-1]))
    return torch.stack(pooled)

import logging
from collections import Counter
from typing import List, Dict, Tuple, Set
from common.data_utils import load_ud_gum_dataset
from config import MIN_WORD_FREQ

logger = logging.getLogger(__name__)

def get_tokenized_corpus(split_name: str = "train") -> List[List[str]]:
    """
    Extract tokenized sentences from UD English GUM split.
    Tokens are lowercased. Sentence boundaries are preserved.
    """
    ds = load_ud_gum_dataset()
    split = ds[split_name]
    sentences = []
    for item in split:
        raw_tokens = item["tokens"]
        # Lowercase tokens consistently
        tokens = [str(t).lower() for t in raw_tokens]
        sentences.append(tokens)
    logger.info(f"Loaded {len(sentences)} sentences for split '{split_name}'.")
    return sentences

def build_vocabulary(sentences: List[List[str]], min_freq: int = MIN_WORD_FREQ) -> Tuple[Dict[str, int], List[str], Counter]:
    """
    Build vocabulary from list of sentences.
    Args:
        sentences: list of tokenized sentences.
        min_freq: minimum frequency threshold.
    Returns:
        word_to_idx: dict mapping word -> index
        idx_to_word: list mapping index -> word
        word_counts: Counter of word frequencies
    """
    counts = Counter()
    for sent in sentences:
        counts.update(sent)

    # Filter by min frequency
    vocab_words = [word for word, count in counts.items() if count >= min_freq]
    # Sort deterministically: by count descending, then alphabetical
    vocab_words.sort(key=lambda w: (-counts[w], w))

    word_to_idx = {word: idx for idx, word in enumerate(vocab_words)}
    idx_to_word = vocab_words

    logger.info(f"Built vocabulary: {len(word_to_idx)} unique words (total tokens: {sum(counts.values())}, min_freq={min_freq}).")
    return word_to_idx, idx_to_word, counts

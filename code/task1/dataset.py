from collections import Counter


def build_vocabulary(
    sentences: list[list[str] | tuple[str, ...]], min_freq: int = 1
) -> tuple[dict[str, int], list[str], Counter[str]]:
    if min_freq < 1:
        raise ValueError("min_freq must be at least 1.")
    counts: Counter[str] = Counter()
    for sentence in sentences:
        counts.update(str(token).lower() for token in sentence)
    vocabulary = sorted(
        (word for word, count in counts.items() if count >= min_freq),
        key=lambda word: (-counts[word], word),
    )
    return {word: index for index, word in enumerate(vocabulary)}, vocabulary, counts

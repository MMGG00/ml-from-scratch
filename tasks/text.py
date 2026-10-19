from collections import Counter

import numpy as np

# Shared tools for any task whose examples are text (reviews, headlines, ...).
# Each text arrives as a list of words, like ["stocks", "fall", "on", "oil", "fears"].


def make_vocab(train_texts: list, vocab_size: int) -> dict:
    # Count how many TRAINING texts each word appears in (set = once per text, sorted = same order every run).
    counts = Counter(word for words in train_texts for word in sorted(set(words)))
    vocab = [word for word, _ in counts.most_common(vocab_size)]   # the most common words, biggest first
    return {word: i for i, word in enumerate(vocab)}               # word -> column number, like {"the": 0, "to": 1}


def to_numbers(texts: list, position: dict) -> np.ndarray:
    # One row per text, one column per vocab word: 1 if the word is in that text, else 0.
    x = np.zeros((len(texts), len(position)))      # (texts, vocab words)
    for row, words in enumerate(texts):
        for word in words:
            if word in position:                   # skip words not in the vocabulary
                x[row, position[word]] = 1
    return x
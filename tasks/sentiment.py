import random
from collections import Counter

import nltk
import numpy as np
from nltk.corpus import movie_reviews

LABELS = ["negative", "positive"]   # position = the number the models see: negative = 0, positive = 1


def to_numbers(reviews: list, position: dict) -> tuple:
    # One row per review, one column per vocabulary word.
    # A cell is 1 if that word appears in that review, else 0.
    x = np.zeros((len(reviews), len(position)))           # (reviews, vocab words)  [1500 x 2000]
    for row, (words, _) in enumerate(reviews):
        for word in words:
            if word in position:                          # skip words not in the vocabulary
                x[row, position[word]] = 1
    y = np.array([label for _, label in reviews])         # one 0 or 1 per review
    return x, y


def load(vocab_size: int = 2000) -> tuple:
    nltk.download("movie_reviews", quiet=True)            # downloads once, then it's saved on your PC

    reviews = []                                          # list of (words, label)
    for file_id in movie_reviews.fileids():               # 2000 reviews, like "neg/cv000_29416.txt"
        words = [word for word in movie_reviews.words(file_id) if word.isalpha()]   # drop , . : and numbers
        label = 0 if file_id.startswith("neg") else 1
        reviews.append((words, label))

    random.seed(0)
    random.shuffle(reviews)
    train_reviews, test_reviews = reviews[:1500], reviews[1500:]

    # Vocabulary: the words found in the most TRAINING reviews. The test reviews stay unseen.
    counts = Counter(word for words, _ in train_reviews for word in sorted(set(words)))
    vocab = [word for word, _ in counts.most_common(vocab_size)]       # ["the", "to", "of", ...]
    position = {word: i for i, word in enumerate(vocab)}               # {"the": 0, "to": 1, ...}

    train_x, train_y = to_numbers(train_reviews, position)
    test_x, test_y = to_numbers(test_reviews, position)
    return train_x, train_y, test_x, test_y

import random

import nltk
import numpy as np
from nltk.corpus import movie_reviews

from tasks.text import make_vocab, to_numbers

LABELS = ["negative", "positive"]   # position = the number the models see: negative = 0, positive = 1


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

    train_words = [words for words, _ in train_reviews]   # just the word lists
    test_words = [words for words, _ in test_reviews]
    position = make_vocab(train_words, vocab_size)        # vocabulary from TRAINING reviews only

    train_x = to_numbers(train_words, position)           # (1500, 2000)
    test_x = to_numbers(test_words, position)             # (500, 2000)
    train_y = np.array([label for _, label in train_reviews])
    test_y = np.array([label for _, label in test_reviews])
    return train_x, train_y, test_x, test_y
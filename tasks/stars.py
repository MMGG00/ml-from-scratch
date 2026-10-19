import csv
import os
import random
import re
import urllib.request

import numpy as np

from tasks.text import make_vocab, to_numbers

# Women's clothing store reviews, each with a 1-5 star rating.
URL = "https://raw.githubusercontent.com/AFAgarap/ecommerce-reviews-analysis/master/Womens%20Clothing%20E-Commerce%20Reviews.csv"
FILE = os.path.join("data", "clothing_reviews.csv")     # where we keep our downloaded copy


def load(vocab_size: int = 2000) -> tuple:
    if not os.path.exists(FILE):                         # download once; after that, read the saved copy
        os.makedirs("data", exist_ok=True)
        urllib.request.urlretrieve(URL, FILE)

    reviews = []                                         # list of (words, label)
    with open(FILE, encoding="utf-8") as f:
        for row in csv.DictReader(f):                    # each row is a dict: column name -> value  {"Rating": "4", ...}
            text = row["Review Text"]
            if not text.strip():                         # some reviews only have stars, no text: skip them
                continue
            words = re.findall(r"[a-z]+", text.lower())  # lowercase, then keep only runs of letters
            reviews.append((words, int(row["Rating"]) - 1))   # stars 1-5 become labels 0-4

    random.seed(0)
    random.shuffle(reviews)
    train_reviews, test_reviews = reviews[:6000], reviews[6000:8000]   # 6000 to learn, 2000 to test

    train_words = [words for words, _ in train_reviews]
    test_words = [words for words, _ in test_reviews]
    position = make_vocab(train_words, vocab_size)

    train_x = to_numbers(train_words, position)          # (6000, 2000)
    test_x = to_numbers(test_words, position)            # (2000, 2000)
    train_y = np.array([label for _, label in train_reviews])
    test_y = np.array([label for _, label in test_reviews])
    return train_x, train_y, test_x, test_y
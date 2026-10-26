import csv
import os
import random
import re
import urllib.request

import numpy as np

from tasks.text import make_vocab, to_numbers

# AG News: real news headlines, each tagged with one of 4 topics.
LABELS = ["World", "Sports", "Business", "Sci/Tech"]   # position = the number the models see
URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"
FILE = os.path.join("data", "ag_news.csv")            # where we keep our downloaded copy


# Model knobs this task wants changed from their defaults:  {model kind: {knob name: value}}.
# Empty means every model uses its normal defaults.
SETTINGS = {}

def load(vocab_size: int = 2000) -> tuple:
    if not os.path.exists(FILE):                       # download once; after that, read the saved copy
        os.makedirs("data", exist_ok=True)
        urllib.request.urlretrieve(URL, FILE)

    headlines = []                                     # list of (words, label)
    with open(FILE, encoding="utf-8") as f:
        for topic, title, _ in csv.reader(f):          # each row: topic number "1"-"4", headline, summary (unused)
            words = re.findall(r"[a-z]+", title.lower())   # lowercase, then keep only runs of letters
            headlines.append((words, int(topic) - 1))      # topics "1"-"4" become 0-3

    random.seed(0)
    random.shuffle(headlines)
    train_headlines, test_headlines = headlines[:6000], headlines[6000:]   # 6000 to learn, 1600 to test

    train_words = [words for words, _ in train_headlines]
    test_words = [words for words, _ in test_headlines]
    position = make_vocab(train_words, vocab_size)

    train_x = to_numbers(train_words, position)        # (6000, 2000)
    test_x = to_numbers(test_words, position)          # (1600, 2000)
    train_y = np.array([label for _, label in train_headlines])
    test_y = np.array([label for _, label in test_headlines])
    return train_x, train_y, test_x, test_y
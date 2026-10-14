import random
from sklearn.datasets import load_digits


def load():
    digits = load_digits()
    all_pixels = digits.data / 16
    all_labels = digits.target
    order = list(range(len(all_pixels)))
    random.seed(0)
    random.shuffle(order)
    train_rows, test_rows = order[:1500], order[1500:]
    return (all_pixels[train_rows], all_labels[train_rows],
            all_pixels[test_rows], all_labels[test_rows])
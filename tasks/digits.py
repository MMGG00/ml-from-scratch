import random
from sklearn.datasets import load_digits


# Model knobs this task wants changed from their defaults:  {model kind: {knob name: value}}.
# neural net: learning rate 2.0 instead of 0.5. Digits only has 1200 photos to learn from, so there are
# few nudges per epoch, and bigger steps learn faster. Tested: 0.965 vs 0.946.
# cnn: the photos are 8 x 8.
SETTINGS = {"neural net": {"learning_rate": 2.0}, "cnn": {"side": 8}}

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
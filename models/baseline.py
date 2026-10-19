import numpy as np


class MostCommon:
    # A "dummy" model: it never looks at the inputs. It always guesses whichever answer
    # was most common in the training data. A real model has to beat this score,
    # or it hasn't learned anything useful.
    #   [stars: always guesses 5 stars and scores 0.555, because 55.5% of test reviews are 5 stars]

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        # train_x is ignored on purpose. Only the labels matter.
        counts = np.bincount(train_y)        # np.ndarray: how many of each label  [stars: [230, 389, 790, 1316, 3275]]
        self.answer = int(counts.argmax())   # int: the label with the biggest count  [stars: 4, which means 5 stars]

    def predict(self, test_x: np.ndarray) -> np.ndarray:
        # np.full(size, value) makes an array of that size filled with one value.
        return np.full(len(test_x), self.answer)   # same guess for every test row  [stars: 2000 copies of 4]
import numpy as np

# Shapes are written as (rows, features).
# rows = how many examples, features = how many numbers describe each one.
# Example in brackets: the digits task, where each picture is 64 pixel numbers.


class KNN:
    def __init__(self, k: int = 3):
        self.k: int = k                          # how many neighbors vote, e.g. 3
        self.train_x: np.ndarray | None = None   # filled by fit: (rows, features)
        self.train_y: np.ndarray | None = None   # filled by fit: one label per row

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        self.train_x = train_x   # (rows, features)  [digits: 1500 x 64]
        self.train_y = train_y   # labels as whole numbers 0, 1, 2, ...  [digits: 2, 8, 7, ...]

    def predict(self, test_x: np.ndarray) -> np.ndarray:
        guesses: list = []
        for point in test_x:                                                # one example: (features,)  [64 numbers]
            distances = np.sqrt(((self.train_x - point) ** 2).sum(axis=1))  # one distance per training row  [1500]
            nearest = np.argsort(distances)[:self.k]                        # row numbers of the k closest  [752, 685, 220]
            votes = self.train_y[nearest]                                   # their labels  [7, 7, 7]
            guesses.append(np.bincount(votes).argmax())                     # most common label  [7]
        return np.array(guesses)                                            # one guess per test row  [297]
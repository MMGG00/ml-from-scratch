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
        # Build the distance grid: one row per new item, one column per known item.
        # Uses (a - b)^2 = a^2 + b^2 - 2ab so NumPy's fast @ does the heavy work.
        test_sq = (test_x ** 2).sum(axis=1)[:, None]                  # a^2 per new item, as a column   [[50],[68]]
        train_sq = (self.train_x ** 2).sum(axis=1)[None, :]           # b^2 per known item, as a row    [[82,68,82]]
        distances = test_sq + train_sq - 2 * test_x @ self.train_x.T  # the grid                        [[4,2,100],[98,72,2]]

        # Same steps as the loop version, done on every row of the grid:
        nearest = np.argsort(distances, axis=1)[:, :self.k]   # each row: row numbers closest first, keep k   k=1: [[1],[2]]
        votes = self.train_y[nearest]                         # row numbers → labels                          [[0],[1]]
        return np.array([np.bincount(row).argmax() for row in votes])   # vote per new item                  [0, 1]                                     # one guess per test row  [297]
import numpy as np

# An ensemble trains several models and lets them vote.
# Each model gives its chances for every answer; we average those chances, then pick the biggest.
# Models with different random starts make different mistakes, so averaging cancels some of them out.
# Numbers in [brackets]: sentiment, 5 neural nets, 500 test reviews, 2 possible answers.


class Ensemble:
    def __init__(self, models: list, seed: int = 0):
        self.models: list = models     # untrained models, e.g. 5 NeuralNets with different seeds  [5]
        self.seed: int = seed          # which batch of seeds this ensemble used (lets compare.py repeat it)

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        for model in self.models:      # each model trains on its own, same data, different random start
            model.fit(train_x, train_y)

    def predict_chances(self, test_x: np.ndarray) -> np.ndarray:
        all_chances = [model.predict_chances(test_x) for model in self.models]   # list of 5 grids, each (rows, classes)  [500 x 2]
        return np.mean(all_chances, axis=0)                                     # average the 5 grids -> one (rows, classes) grid

    def predict(self, test_x: np.ndarray) -> np.ndarray:
        return self.predict_chances(test_x).argmax(axis=1)   # one guess per row  [500]
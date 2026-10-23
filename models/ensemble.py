import numpy as np

# An ensemble trains several models and lets them answer as a team.
# Each model gives its chances for every answer; we average those chances, then pick the biggest.
# Models with different random starts make different mistakes, so averaging cancels some of them out.
# Numbers in [brackets]: fashion, 3 neural nets, 2000 test photos, 10 possible answers.
#
# No seed is stored on purpose: compare.py runs models WITHOUT a seed only once. The team already averages
# away its own luck, so repeating it 3 times (9 nets) mostly measured that its spread is small.


class Ensemble:
    def __init__(self, models: list):
        self.models: list = models     # untrained models, e.g. 3 NeuralNets with different seeds  [3]

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        for model in self.models:      # each model trains on its own, same data, different random start
            model.fit(train_x, train_y)

    def predict_chances(self, test_x: np.ndarray) -> np.ndarray:
        all_chances = [model.predict_chances(test_x) for model in self.models]   # list of 3 grids, each (rows, classes)  [2000 x 10]
        return np.mean(all_chances, axis=0)                                     # average the 3 grids -> one (rows, classes) grid

    def predict(self, test_x: np.ndarray) -> np.ndarray:
        return self.predict_chances(test_x).argmax(axis=1)   # one guess per row  [2000]
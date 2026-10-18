import numpy as np

from models.neural_net import softmax

# Logistic regression = a neural net with NO hidden layer.
# The inputs go straight to one score per possible answer.
# Numbers in [brackets] are examples from our two tasks:
#   digits:    1500 pictures, 64 pixels each, 10 possible answers
#   sentiment: 1500 reviews, 2000 vocab words each, 2 possible answers


class LogisticRegression:
    def __init__(self, learning_rate: float = 1.0, steps: int = 1000):
        self.learning_rate: float = learning_rate     # how big each move is  [1.0]
        self.steps: int = steps                       # how many training rounds  [1000]
        self.weights: np.ndarray | None = None        # filled by fit: (features, classes)  [64 x 10] or [2000 x 2]
        self.biases: np.ndarray | None = None         # filled by fit: (classes,)  [10] or [2]

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        n = len(train_x)                              # how many examples  [1500]
        features = train_x.shape[1]                   # numbers per example  [64] or [2000]
        classes = int(train_y.max()) + 1              # how many possible answers  [10] or [2]
        answers = np.eye(classes)[train_y]            # one-hot: (rows, classes)

        self.weights = np.zeros((features, classes))  # every weight starts at 0
        self.biases = np.zeros(classes)

        for step in range(self.steps):
            scores = train_x @ self.weights + self.biases    # (rows, classes)
            chances = softmax(scores)                        # (rows, classes), each row adds to 1
            errors = (chances - answers) / n                 # how wrong, averaged over rows
            weight_slopes = train_x.T @ errors               # (features, classes)
            bias_slopes = errors.sum(axis=0)                 # (classes,)
            self.weights -= self.learning_rate * weight_slopes
            self.biases -= self.learning_rate * bias_slopes

    def predict(self, test_x: np.ndarray) -> np.ndarray:
        scores = test_x @ self.weights + self.biases        # (rows, classes)
        return softmax(scores).argmax(axis=1)               # one guess per row
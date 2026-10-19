import numpy as np

from models.neural_net import softmax

# Logistic regression = a neural net with NO hidden layer.
# The inputs go straight to one score per possible answer.
# Numbers in [brackets] are examples from our two tasks:
#   digits:    1500 pictures (1200 learn + 300 check), 64 pixels each, 10 possible answers
#   sentiment: 1500 reviews (1200 learn + 300 check), 2000 vocab words each, 2 possible answers


class LogisticRegression:
    def __init__(self, learning_rate: float = 1.0, max_steps: int = 5000,
                 check_every: int = 10, patience: int = 20, check_fraction: float = 0.2):
        self.learning_rate: float = learning_rate     # how big each move is  [1.0]
        self.max_steps: int = max_steps               # hard cap on training rounds  [5000]
        self.check_every: int = check_every           # quiz the check pile every this many steps  [10]
        self.patience: int = patience                 # quizzes in a row with no new best before stopping  [20]
        self.check_fraction: float = check_fraction   # share of the data set aside as the check pile  [0.2]
        self.weights: np.ndarray | None = None        # filled by fit: (features, classes)  [64 x 10] or [2000 x 2]
        self.biases: np.ndarray | None = None         # filled by fit: (classes,)  [10] or [2]
        self.history: list = []                       # (step, quiz score) at every check, for charts
        self.best_step: int = 0                       # the step whose knobs we kept

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        # 1. Split off a check pile (same as the neural net).
        n_check = int(len(train_x) * self.check_fraction)        # [300]
        check_x, check_y = train_x[-n_check:], train_y[-n_check:]
        train_x, train_y = train_x[:-n_check], train_y[:-n_check]

        n = len(train_x)                              # how many examples it learns from  [1200]
        features = train_x.shape[1]                   # numbers per example  [64] or [2000]
        classes = int(train_y.max()) + 1              # how many possible answers  [10] or [2]
        answers = np.eye(classes)[train_y]            # one-hot: (rows, classes)

        self.weights = np.zeros((features, classes))  # every weight starts at 0
        self.biases = np.zeros(classes)

        # 2. Early stopping bookkeeping (same as the neural net, but only 2 knob arrays to save).
        best_score = -1.0
        best_knobs = None
        checks_without_better = 0
        self.history = []

        for step in range(self.max_steps):
            # 3. Quiz, maybe save, maybe stop.
            if step % self.check_every == 0:
                score = float(np.mean(self.predict(check_x) == check_y))
                self.history.append((step, score))
                if score > best_score:
                    best_score = score
                    self.best_step = step
                    best_knobs = (self.weights.copy(), self.biases.copy())
                    checks_without_better = 0
                else:
                    checks_without_better += 1
                    if checks_without_better >= self.patience:
                        break

            # 4. The usual training step (unchanged).
            scores = train_x @ self.weights + self.biases    # (rows, classes)
            chances = softmax(scores)                        # (rows, classes), each row adds to 1
            errors = (chances - answers) / n                 # how wrong, averaged over rows
            weight_slopes = train_x.T @ errors               # (features, classes)
            bias_slopes = errors.sum(axis=0)                 # (classes,)
            self.weights -= self.learning_rate * weight_slopes
            self.biases -= self.learning_rate * bias_slopes

        # 5. Go back to the best knobs.
        self.weights, self.biases = best_knobs

    def predict(self, test_x: np.ndarray) -> np.ndarray:
        scores = test_x @ self.weights + self.biases        # (rows, classes)
        return softmax(scores).argmax(axis=1)               # one guess per row
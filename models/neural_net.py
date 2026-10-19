import numpy as np

# Shapes are written as (rows, columns).
# Numbers in [brackets] are the digits task, as an example:
#   1500 = examples given to fit, split into 1200 to learn from + 300 to check on,
#   64 = pixels per picture (8x8 grid), 10 = possible answers (digits 0-9),
#   32 = hidden neurons (hidden_size, our choice).
# Another task gives different numbers; the code reads them from the data.

def squash(totals: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-totals))                  # any number -> between 0 and 1


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = scores - scores.max(axis=1, keepdims=True)   # subtract each row's biggest score (stops overflow)
    exps = np.exp(scores)
    return exps / exps.sum(axis=1, keepdims=True)          # each row now adds up to 1


class NeuralNet:
    def __init__(self, hidden_size: int = 32, learning_rate: float = 2.0, max_steps: int = 5000,
                 check_every: int = 10, patience: int = 20, check_fraction: float = 0.2, seed: int = 0):
        self.hidden_size: int = hidden_size            # neurons in the hidden layer  [32]
        self.learning_rate: float = learning_rate      # how big each move is  [2.0]
        self.max_steps: int = max_steps                # hard cap on training rounds  [5000]
        self.check_every: int = check_every            # quiz the check pile every this many steps  [10]
        self.patience: int = patience                  # quizzes in a row with no new best before stopping  [20]
        self.check_fraction: float = check_fraction    # share of the data set aside as the check pile  [0.2]
        self.seed: int = seed                          # same random start every run
        self.hidden_weights: np.ndarray | None = None  # filled by fit: (features, hidden_size)  [64 x 32]
        self.hidden_biases: np.ndarray | None = None   # filled by fit: (hidden_size,)  [32]
        self.output_weights: np.ndarray | None = None  # filled by fit: (hidden_size, classes)  [32 x 10]
        self.output_biases: np.ndarray | None = None   # filled by fit: (classes,)  [10]
        self.history: list = []                        # (step, quiz score) at every check, for charts
        self.best_step: int = 0                        # the step whose knobs we kept

    def forward(self, x: np.ndarray) -> tuple:
        hidden_out = squash(x @ self.hidden_weights + self.hidden_biases)   # (rows, hidden_size)  [1200 x 32]
        scores = hidden_out @ self.output_weights + self.output_biases      # (rows, classes)  [1200 x 10]
        chances = softmax(scores)                                           # (rows, classes), rows add to 1
        return hidden_out, chances

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        # 1. Split off a check pile. The task already shuffled the data, so the last rows are a fair mix.
        n_check = int(len(train_x) * self.check_fraction)        # [300]
        check_x, check_y = train_x[-n_check:], train_y[-n_check:]  # the quiz: only looked at, never learned from
        train_x, train_y = train_x[:-n_check], train_y[:-n_check]  # what the knobs learn from  [1200]

        n = len(train_x)                                   # how many examples it learns from  [1200]
        features = train_x.shape[1]                        # numbers per example  [64]
        classes = int(train_y.max()) + 1                   # how many possible answers  [10]
        answers = np.eye(classes)[train_y]                 # one-hot: (rows, classes)  [1200 x 10]

        rng = np.random.default_rng(self.seed)
        self.hidden_weights = rng.uniform(-0.1, 0.1, (features, self.hidden_size))
        self.hidden_biases = np.zeros(self.hidden_size)
        self.output_weights = rng.uniform(-0.1, 0.1, (self.hidden_size, classes))
        self.output_biases = np.zeros(classes)

        # 2. Early stopping bookkeeping.
        best_score = -1.0                                  # best quiz score so far (starts below any real score)
        best_knobs = None                                  # copies of the 4 knob arrays at that best moment
        checks_without_better = 0                          # how many quizzes in a row didn't beat the best
        self.history = []

        for step in range(self.max_steps):
            # 3. Every check_every steps: quiz, maybe save, maybe stop.
            if step % self.check_every == 0:
                score = float(np.mean(self.predict(check_x) == check_y))   # fraction right on the check pile
                self.history.append((step, score))
                if score > best_score:                     # new best: remember it
                    best_score = score
                    self.best_step = step
                    best_knobs = (self.hidden_weights.copy(), self.hidden_biases.copy(),
                                  self.output_weights.copy(), self.output_biases.copy())
                    checks_without_better = 0
                else:                                      # no improvement this time
                    checks_without_better += 1
                    if checks_without_better >= self.patience:
                        break                              # waited long enough: stop training

            # 4. The usual training step (same as before), only on the learn pile.
            hidden_out, chances = self.forward(train_x)
            errors = (chances - answers) / n
            output_weight_slopes = hidden_out.T @ errors
            output_bias_slopes = errors.sum(axis=0)
            blame = (errors @ self.output_weights.T) * hidden_out * (1 - hidden_out)
            hidden_weight_slopes = train_x.T @ blame
            hidden_bias_slopes = blame.sum(axis=0)

            self.hidden_weights -= self.learning_rate * hidden_weight_slopes
            self.hidden_biases -= self.learning_rate * hidden_bias_slopes
            self.output_weights -= self.learning_rate * output_weight_slopes
            self.output_biases -= self.learning_rate * output_bias_slopes

        # 5. Go back to the best knobs we saw, not whatever the last step left behind.
        self.hidden_weights, self.hidden_biases, self.output_weights, self.output_biases = best_knobs

    def predict(self, test_x: np.ndarray) -> np.ndarray:
        _, chances = self.forward(test_x)                  # (rows, classes)  [297 x 10]
        return chances.argmax(axis=1)                      # one guess per row  [297]
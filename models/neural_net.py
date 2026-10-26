import numpy as np


# Shapes are written as (rows, columns).
# Numbers in [brackets] are the digits task, as an example:
#   1500 = examples given to fit, split into 1200 to learn from + 300 to check on,
#   64 = pixels per picture (8x8 grid), 10 = possible answers (digits 0-9),
#   32 = hidden neurons (hidden_size, our choice).
# Another task gives different numbers; the code reads them from the data.

def squash(totals: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-totals))  # any number -> between 0 and 1


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = scores - scores.max(axis=1, keepdims=True)  # subtract each row's biggest score (stops overflow)
    exps = np.exp(scores)
    return exps / exps.sum(axis=1, keepdims=True)  # each row now adds up to 1


class NeuralNet:
    def __init__(self, hidden_size: int = 32, learning_rate: float = 0.5, batch_size: int = 50,
                 max_epochs: int = 100, patience: int = 10, check_fraction: float = 0.2, seed: int = 0):
        self.hidden_size: int = hidden_size  # neurons in the hidden layer  [32]
        self.learning_rate: float = learning_rate  # how big each move is  [0.5]
        self.batch_size: int = batch_size  # examples per nudge  [50]
        self.max_epochs: int = max_epochs  # most passes through all the examples  [100]
        self.patience: int = patience  # epochs in a row with no new best before stopping  [10]
        self.check_fraction: float = check_fraction  # share of the data set aside as the check pile  [0.2]
        self.seed: int = seed  # same random start every run
        self.hidden_weights: np.ndarray | None = None  # filled by fit: (features, hidden_size)  [64 x 32]
        self.hidden_biases: np.ndarray | None = None  # filled by fit: (hidden_size,)  [32]
        self.output_weights: np.ndarray | None = None  # filled by fit: (hidden_size, classes)  [32 x 10]
        self.output_biases: np.ndarray | None = None  # filled by fit: (classes,)  [10]
        self.history: list = []  # (epoch, quiz score) after every epoch, for charts
        self.best_epoch: int = 0  # the epoch whose knobs we kept

    def forward(self, x: np.ndarray) -> tuple:
        hidden_out = squash(x @ self.hidden_weights + self.hidden_biases)  # (rows, hidden_size)  [50 x 32]
        scores = hidden_out @ self.output_weights + self.output_biases  # (rows, classes)  [50 x 10]
        chances = softmax(scores)  # (rows, classes), rows add to 1
        return hidden_out, chances

    def nudge(self, batch_x: np.ndarray, batch_answers: np.ndarray) -> None:
        # One learning step from one group of examples: guess, measure how wrong, move every knob a little.
        hidden_out, chances = self.forward(batch_x)
        errors = (chances - batch_answers) / len(batch_x)  # how wrong, averaged over the group  [50 x 10]
        output_weight_slopes = hidden_out.T @ errors
        output_bias_slopes = errors.sum(axis=0)
        blame = (errors @ self.output_weights.T) * hidden_out * (1 - hidden_out)
        hidden_weight_slopes = batch_x.T @ blame
        hidden_bias_slopes = blame.sum(axis=0)

        self.hidden_weights -= self.learning_rate * hidden_weight_slopes
        self.hidden_biases -= self.learning_rate * hidden_bias_slopes
        self.output_weights -= self.learning_rate * output_weight_slopes
        self.output_biases -= self.learning_rate * output_bias_slopes

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        # Split off a check pile. The task already shuffled the data, so the last rows are a fair mix.
        n_check = int(len(train_x) * self.check_fraction)  # [300]
        check_x, check_y = train_x[-n_check:], train_y[-n_check:]  # the quiz: only looked at, never learned from
        train_x, train_y = train_x[:-n_check], train_y[:-n_check]  # what the knobs learn from  [1200]

        features = train_x.shape[1]  # numbers per example  [64]
        classes = int(train_y.max()) + 1  # how many possible answers  [10]
        answers = np.eye(classes)[train_y]  # one-hot: (rows, classes)  [1200 x 10]

        rng = np.random.default_rng(self.seed)
        self.hidden_weights = rng.uniform(-0.1, 0.1, (features, self.hidden_size))
        self.hidden_biases = np.zeros(self.hidden_size)
        self.output_weights = rng.uniform(-0.1, 0.1, (self.hidden_size, classes))
        self.output_biases = np.zeros(classes)

        best_score = -1.0  # best quiz score so far (starts below any real score)
        best_knobs = None  # copies of the 4 knob arrays at that best moment
        epochs_without_better = 0  # how many epochs in a row didn't beat the best
        self.history = []

        for epoch in range(self.max_epochs):
            # One pass through every example, in a new shuffled order, one group at a time.
            order = rng.permutation(len(train_x))
            for start in range(0, len(train_x), self.batch_size):
                batch = order[start:start + self.batch_size]
                self.nudge(train_x[batch], answers[batch])

            # After each pass: quiz, maybe save, maybe stop.
            score = float(np.mean(self.predict(check_x) == check_y))  # fraction right on the check pile
            self.history.append((epoch, score))
            if score > best_score:
                best_score = score
                self.best_epoch = epoch
                best_knobs = (self.hidden_weights.copy(), self.hidden_biases.copy(),
                              self.output_weights.copy(), self.output_biases.copy())
                epochs_without_better = 0
            else:
                epochs_without_better += 1
                if epochs_without_better >= self.patience:
                    break

        # Go back to the best knobs we saw, not whatever the last epoch left behind.
        self.hidden_weights, self.hidden_biases, self.output_weights, self.output_biases = best_knobs

    def predict_chances(self, test_x: np.ndarray) -> np.ndarray:
        _, chances = self.forward(test_x)  # (rows, classes)  [297 x 10], each row adds to 1
        return chances

    def predict(self, test_x: np.ndarray) -> np.ndarray:
        return self.predict_chances(test_x).argmax(axis=1)  # one guess per row  [297]
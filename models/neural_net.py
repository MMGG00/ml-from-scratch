import numpy as np

# Shapes are written as (rows, columns).
# Numbers in [brackets] are the digits task, as an example:
#   1500 = training pictures, 297 = test pictures,
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
    def __init__(self, hidden_size: int = 32, learning_rate: float = 2.0, steps: int = 1000, seed: int = 0):
        self.hidden_size: int = hidden_size            # neurons in the hidden layer  [32]
        self.learning_rate: float = learning_rate      # how big each move is  [2.0]
        self.steps: int = steps                        # how many training rounds  [1000]
        self.seed: int = seed                          # same random start every run
        self.hidden_weights: np.ndarray | None = None  # filled by fit: (features, hidden_size)  [64 x 32]
        self.hidden_biases: np.ndarray | None = None   # filled by fit: (hidden_size,)  [32]
        self.output_weights: np.ndarray | None = None  # filled by fit: (hidden_size, classes)  [32 x 10]
        self.output_biases: np.ndarray | None = None   # filled by fit: (classes,)  [10]
        self.costs: list = []                          # cost every 100 steps, to see it go down

    def forward(self, x: np.ndarray) -> tuple:
        hidden_out = squash(x @ self.hidden_weights + self.hidden_biases)   # (rows, hidden_size)  [1500 x 32]
        scores = hidden_out @ self.output_weights + self.output_biases      # (rows, classes)  [1500 x 10]
        chances = softmax(scores)                                           # (rows, classes), rows add to 1
        return hidden_out, chances

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        n = len(train_x)                                   # how many examples  [1500]
        features = train_x.shape[1]                        # numbers per example  [64]
        classes = int(train_y.max()) + 1                   # how many possible answers  [10]
        answers = np.eye(classes)[train_y]                 # one-hot: (rows, classes)  [1500 x 10]

        rng = np.random.default_rng(self.seed)
        self.hidden_weights = rng.uniform(-0.1, 0.1, (features, self.hidden_size))
        self.hidden_biases = np.zeros(self.hidden_size)
        self.output_weights = rng.uniform(-0.1, 0.1, (self.hidden_size, classes))
        self.output_biases = np.zeros(classes)

        for step in range(self.steps):
            hidden_out, chances = self.forward(train_x)

            errors = (chances - answers) / n                                      # how wrong, averaged  [1500 x 10]
            output_weight_slopes = hidden_out.T @ errors                          # [32 x 10]
            output_bias_slopes = errors.sum(axis=0)                               # [10]
            blame = (errors @ self.output_weights.T) * hidden_out * (1 - hidden_out)   # [1500 x 32]
            hidden_weight_slopes = train_x.T @ blame                              # [64 x 32]
            hidden_bias_slopes = blame.sum(axis=0)                                # [32]

            self.hidden_weights -= self.learning_rate * hidden_weight_slopes
            self.hidden_biases -= self.learning_rate * hidden_bias_slopes
            self.output_weights -= self.learning_rate * output_weight_slopes
            self.output_biases -= self.learning_rate * output_bias_slopes

            if step % 100 == 0:
                right_answer_chances = chances[np.arange(n), train_y]   # chance it gave the right answer, per row
                self.costs.append(float(-np.mean(np.log(right_answer_chances))))

    def predict(self, test_x: np.ndarray) -> np.ndarray:
        _, chances = self.forward(test_x)                  # (rows, classes)  [297 x 10]
        return chances.argmax(axis=1)                      # one guess per row  [297]
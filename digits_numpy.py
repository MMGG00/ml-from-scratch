import random
import numpy as np
from sklearn.datasets import load_digits


def squash(totals):
    return 1 / (1 + np.exp(-totals))


def softmax(scores):
    # One row per picture, 10 scores per row -> 10 chances per row that add up to 1.
    scores = scores - scores.max(axis=1, keepdims=True)  # stops exp() from overflowing
    exps = np.exp(scores)
    return exps / exps.sum(axis=1, keepdims=True)


def forward(pixels, hidden_weights, hidden_biases, output_weights, output_biases):
    hidden_out = squash(pixels @ hidden_weights + hidden_biases)   # every picture x every hidden neuron
    scores = hidden_out @ output_weights + output_biases           # every picture x 10 digit scores
    chances = softmax(scores)                                      # every picture x 10 chances
    return hidden_out, chances


def train(pixels, answers, hidden_size=32, learning_rate=2.0, steps=1000, seed=0):
    rng = np.random.default_rng(seed)
    hidden_weights = rng.uniform(-0.1, 0.1, (pixels.shape[1], hidden_size))  # 64 x 32
    hidden_biases = np.zeros(hidden_size)                                    # 32
    output_weights = rng.uniform(-0.1, 0.1, (hidden_size, 10))               # 32 x 10
    output_biases = np.zeros(10)                                             # 10
    n = len(pixels)

    for step in range(steps):
        hidden_out, chances = forward(pixels, hidden_weights, hidden_biases,
                                      output_weights, output_biases)

        # Backprop: same gears as network.py, for every picture at once.
        errors = (chances - answers) / n                                   # how wrong (averaged)
        output_weight_slopes = hidden_out.T @ errors                       # error x hidden_out
        output_bias_slopes = errors.sum(axis=0)                            # just the error
        blame = (errors @ output_weights.T) * hidden_out * (1 - hidden_out)  # error x out weight x steepness
        hidden_weight_slopes = pixels.T @ blame                            # blame x pixel
        hidden_bias_slopes = blame.sum(axis=0)                             # just the blame

        # Move every knob downhill.
        hidden_weights -= learning_rate * hidden_weight_slopes
        hidden_biases -= learning_rate * hidden_bias_slopes
        output_weights -= learning_rate * output_weight_slopes
        output_biases -= learning_rate * output_bias_slopes

        if step % 100 == 0:
            right_answer_chances = chances[np.arange(n), answers.argmax(axis=1)]
            cost = -np.mean(np.log(right_answer_chances))
            print("step", step, "cost", round(float(cost), 4))
    return hidden_weights, hidden_biases, output_weights, output_biases


def accuracy(pixels, labels, hidden_weights, hidden_biases, output_weights, output_biases):
    _, chances = forward(pixels, hidden_weights, hidden_biases, output_weights, output_biases)
    guesses = chances.argmax(axis=1)            # position of the biggest chance = the guessed digit
    return float(np.mean(guesses == labels))


if __name__ == "__main__":
    digits = load_digits()
    all_pixels = digits.data / 16      # 1797 x 64, each row is one picture, shrunk to 0-1
    all_labels = digits.target         # 1797 digits, 0-9

    # Same shuffle as digits.py, so the same 297 test pictures as kNN.
    order = list(range(len(all_pixels)))
    random.seed(0)
    random.shuffle(order)
    train_rows, test_rows = order[:1500], order[1500:]

    train_pixels, test_pixels = all_pixels[train_rows], all_pixels[test_rows]
    train_labels, test_labels = all_labels[train_rows], all_labels[test_rows]
    train_answers = np.eye(10)[train_labels]   # one-hot: 1500 x 10

    hidden_weights, hidden_biases, output_weights, output_biases = train(train_pixels, train_answers)
    score = accuracy(test_pixels, test_labels, hidden_weights, hidden_biases, output_weights, output_biases)
    print("test accuracy", round(score, 3))
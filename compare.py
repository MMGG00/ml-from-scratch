import time
import numpy as np
from models.knn import KNN
from models.neural_net import NeuralNet
from models.logistic import LogisticRegression
from tasks import digits, sentiment

# Every task: (name, its load function). Note: digits.load with no (), we hand over the function itself.
TASKS = [
    ("Guess Digits", digits.load),
    ("Sentiment", sentiment.load),
]

SEEDS = [0, 1, 2, 3, 4]   # random starts to try, for models that have randomness


def make_models(seed: int) -> list:
    # Fresh, untrained models every time. Only the neural net uses the seed;
    # kNN and logistic have no randomness, so they ignore it.
    return [
        ("kNN k=1", KNN(k=1)),
        ("kNN k=3", KNN(k=3)),
        ("kNN k=5", KNN(k=5)),
        ("neural net", NeuralNet(seed=seed)),
        ("logistic", LogisticRegression()),
    ]


if __name__ == "__main__":
    print(f"{'task':<16}{'model':<14}{'avg acc':<10}{'spread':<10}{'runs':<6}{'train s':<10}{'guess s':<10}")
    for task_name, load in TASKS:
        train_x, train_y, test_x, test_y = load()            # same data for every model

        for i, (model_name, _) in enumerate(make_models(0)):   # i = which model in the list
            scores, train_times, guess_times = [], [], []
            for seed in SEEDS:
                model = make_models(seed)[i][1]               # a fresh copy of model i, built with this seed

                start = time.time()
                model.fit(train_x, train_y)
                train_times.append(time.time() - start)

                start = time.time()
                guesses = model.predict(test_x)
                guess_times.append(time.time() - start)

                scores.append(float(np.mean(guesses == test_y)))
                if not hasattr(model, "seed"):                # no randomness: every run would be identical
                    break

            spread = max(scores) - min(scores)                # best run minus worst run
            print(f"{task_name:<16}{model_name:<14}{np.mean(scores):<10.3f}{spread:<10.3f}{len(scores):<6}"
                  f"{np.mean(train_times):<10.2f}{np.mean(guess_times):<10.2f}")
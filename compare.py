import time
import numpy as np
from tasks import digits
from models.knn import KNN
from models.neural_net import NeuralNet

# Every task: (name, its load function). Note: digits.load with no (), we hand over the function itself.
TASKS = [
    ("Guess Digits ", digits.load),
]


def make_models() -> list:
    # Fresh, untrained models every time, so nothing carries over from the last task.
    return [
        ("kNN k=1", KNN(k=1)),
        ("kNN k=3", KNN(k=3)),
        ("kNN k=5", KNN(k=5)),
        ("neural net", NeuralNet()),
    ]


if __name__ == "__main__":
    print(f"{'task':<16}{'model':<14}{'accuracy':<10}{'train s':<10}{'guess s':<10}")
    for task_name, load in TASKS:
        train_x, train_y, test_x, test_y = load()            # same data for every model

        for model_name, model in make_models():
            start = time.time()
            model.fit(train_x, train_y)                      # learn (or, for kNN, memorize)
            train_seconds = time.time() - start

            start = time.time()
            guesses = model.predict(test_x)                  # one guess per test example
            guess_seconds = time.time() - start

            accuracy = float(np.mean(guesses == test_y))     # fraction right
            print(f"{task_name:<16}{model_name:<14}{accuracy:<10.3f}{train_seconds:<10.2f}{guess_seconds:<10.2f}")
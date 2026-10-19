import sys                                   # NEW: lets us read what you typed in the terminal
import time
import numpy as np
from models.knn import KNN
from models.neural_net import NeuralNet
from models.logistic import LogisticRegression
from models.ensemble import Ensemble
from tasks import digits, sentiment, headlines   # NEW: added headlines

# Every task: (name, its load function). digits.load has no (), so we hand over the function itself.
TASKS = [
    ("digits", digits.load),                 # NEW: one-word names
    ("sentiment", sentiment.load),
    ("headlines", headlines.load),           # NEW
]

SEEDS = [0, 1, 2]                            # NEW: was 5 seeds; 3 keeps headlines from taking forever


def make_models(seed: int) -> list:
    return [
        ("kNN k=1", KNN(k=1)),
        ("kNN k=3", KNN(k=3)),
        ("kNN k=5", KNN(k=5)),
        ("neural net", NeuralNet(seed=seed)),
        ("logistic", LogisticRegression()),
        ("ensemble x3", Ensemble([NeuralNet(seed=seed * 3 + k) for k in range(3)], seed=seed)),
    ]


if __name__ == "__main__":
    chosen = sys.argv[1:]                    # NEW: list[str], task names you typed  [["headlines"]] or []

    print(f"{'task':<12}{'model':<14}{'avg acc':<10}{'spread':<10}{'runs':<6}{'train s':<10}{'guess s':<10}")

    for task_name, load in TASKS:
        if chosen and task_name not in chosen:   # NEW: you named tasks, and this isn't one of them...
            continue                             # NEW: ...so skip to the next task
        train_x, train_y, test_x, test_y = load()

        for i, (model_name, _) in enumerate(make_models(0)):
            scores, train_times, guess_times = [], [], []
            for seed in SEEDS:
                model = make_models(seed)[i][1]

                start = time.time()
                model.fit(train_x, train_y)
                train_times.append(time.time() - start)

                start = time.time()
                guesses = model.predict(test_x)
                guess_times.append(time.time() - start)

                scores.append(float(np.mean(guesses == test_y)))
                if not hasattr(model, "seed"):
                    break

            spread = max(scores) - min(scores)
            print(f"{task_name:<12}{model_name:<14}{np.mean(scores):<10.3f}{spread:<10.3f}{len(scores):<6}"
                  f"{np.mean(train_times):<10.2f}{np.mean(guess_times):<10.2f}")
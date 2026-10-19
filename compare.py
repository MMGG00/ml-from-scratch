import sys                      # lets us read the words typed after "python compare.py"
import time                     # stopwatch: time.time() = seconds since 1970, so (end - start) = seconds taken
import numpy as np
from models.knn import KNN
from models.neural_net import NeuralNet
from models.logistic import LogisticRegression
from models.ensemble import Ensemble
from models.baseline import MostCommon
from tasks import digits, sentiment, headlines, stars

# Every task: (name, load function). No () after load, so we store the function itself and call it later.
# Each load() returns 4 arrays:  train_x (rows, features)  train_y (rows,)  test_x  test_y
#   [digits: train_x (1500, 64), train_y (1500,) like [2, 8, 7, ...]]
TASKS = [
    ("digits", digits.load),          # guess 0-9 from an 8x8 pixel image
    ("sentiment", sentiment.load),    # guess bad (0) or good (1) from a movie review
    ("headlines", headlines.load),    # guess World / Sports / Business / Sci-Tech from a news headline
    ("stars", stars.load),            # guess 1-5 stars (labels 0-4) from a clothing review
]

SEEDS = [0, 1, 2]   # list[int]: random starts to try, for models that use randomness


def make_models(seed: int) -> list:
    # Returns a list of (name, model) pairs, all fresh and untrained.  [("kNN k=1", KNN object), ...]
    # The neural net and ensemble use the seed for their random starting weights;
    # most common, kNN and logistic have no randomness, so they don't take one.
    return [
        ("most common", MostCommon()),   # dummy: always guesses the most common training answer; every model should beat it
        ("kNN k=1", KNN(k=1)),
        ("kNN k=3", KNN(k=3)),
        ("kNN k=5", KNN(k=5)),
        ("neural net", NeuralNet(seed=seed)),
        ("logistic", LogisticRegression()),
        # 3 neural nets averaged together. Seeds: seed 0 -> nets 0,1,2 / seed 1 -> 3,4,5 / seed 2 -> 6,7,8
        ("ensemble x3", Ensemble([NeuralNet(seed=seed * 3 + k) for k in range(3)], seed=seed)),
    ]


if __name__ == "__main__":     # only runs when you start this file directly, not when another file imports it
    # sys.argv = list[str] of everything typed:  "python compare.py stars"  ->  ["compare.py", "stars"]
    # [1:] drops the file name, leaving the task names:  ["stars"]   (typed nothing extra -> [])
    chosen = sys.argv[1:]

    # Header row. :<12 means "left-align in a space 12 characters wide" so the columns line up.
    print(f"{'task':<12}{'model':<14}{'avg acc':<10}{'spread':<10}{'runs':<6}{'train s':<10}{'guess s':<10}")

    for task_name, load in TASKS:                  # task_name: str ["stars"], load: function [stars.load]
        # Skip this task if you named tasks and this isn't one of them. An empty list counts as False,
        # so typing no names means nothing gets skipped.
        if chosen and task_name not in chosen:
            continue                               # jump to the next task in TASKS

        train_x, train_y, test_x, test_y = load()  # same data for every model  [stars: (6000, 2000), (6000,), (2000, 2000), (2000,)]

        # enumerate gives (position, item):  i = 0, 1, 2 ...   model_name = "kNN k=1", ...
        # make_models(0) is only used here to get the names; _ means "ignore this part" (the model object).
        for i, (model_name, _) in enumerate(make_models(0)):
            scores = []          # list[float]: accuracy of each run   [digits neural net: 0.963, 0.966, 0.963]
            train_times = []     # list[float]: seconds fit took, each run
            guess_times = []     # list[float]: seconds predict took, each run

            for seed in SEEDS:
                model = make_models(seed)[i][1]    # [i] = the i-th (name, model) pair, [1] = the model part

                start = time.time()
                model.fit(train_x, train_y)        # learn from the training rows
                train_times.append(time.time() - start)

                start = time.time()
                guesses = model.predict(test_x)    # np.ndarray of ints, one guess per test row  [digits: 297 guesses like 7, 3, 3, 5, 3, ...]
                guess_times.append(time.time() - start)

                # guesses == test_y -> array of True/False; mean counts True as 1, False as 0 -> fraction right
                #   guesses [7,3,3] vs answers [7,3,5] -> [True, True, False] -> 2/3 = 0.667
                scores.append(float(np.mean(guesses == test_y)))

                if not hasattr(model, "seed"):     # no seed = no randomness = every run would be identical
                    break                          # so 1 run is enough (that's why most common/kNN/logistic show runs = 1)

            spread = max(scores) - min(scores)     # best run minus worst run: how much luck matters  [0.966 - 0.963 = 0.003]
            # Times are the AVERAGE for one run, not the total.
            print(f"{task_name:<12}{model_name:<14}{np.mean(scores):<10.3f}{spread:<10.3f}{len(scores):<6}"
                  f"{np.mean(train_times):<10.2f}{np.mean(guess_times):<10.2f}")
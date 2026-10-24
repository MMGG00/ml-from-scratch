import csv                      # reads and writes .csv files (rows split by commas)
import os                       # file tools; here, checking whether a file already exists
import sys                      # lets us read the words typed after "python compare.py"
import time                     # stopwatch: time.time() = seconds since 1970, so (end - start) = seconds taken
from datetime import datetime   # today's date and time, to stamp each run
import numpy as np
from models.knn import KNN
from models.neural_net import NeuralNet
from models.logistic import LogisticRegression
from models.ensemble import Ensemble
from models.baseline import MostCommon
from tasks import digits, sentiment, headlines, stars, fashion

# Every task: (name, load function, settings).
#   load: no () after it, so we store the function itself and call it later.
#   settings: the task's SETTINGS dict, the model knobs it wants changed  [fashion: {"logistic": {"learning_rate": 0.1}}]
# Each load() returns 4 arrays:  train_x (rows, features)  train_y (rows,)  test_x  test_y
#   [digits: train_x (1500, 64), train_y (1500,) like [2, 8, 7, ...]]
TASKS = [
    ("digits", digits.load, digits.SETTINGS),           # guess 0-9 from an 8x8 pixel image
    ("sentiment", sentiment.load, sentiment.SETTINGS),  # guess bad (0) or good (1) from a movie review
    ("headlines", headlines.load, headlines.SETTINGS),  # guess World / Sports / Business / Sci-Tech from a news headline
    ("stars", stars.load, stars.SETTINGS),              # guess 1-5 stars (labels 0-4) from a clothing review
    ("fashion", fashion.load, fashion.SETTINGS),        # guess the clothing type (shirt, shoe, bag ...) from a 28x28 photo
]

SEEDS = [0, 1, 2]   # list[int]: random starts to try, for models that use randomness

# Every printed row also gets saved here, so past results never get lost.
# Column names are lowercase with no spaces, the way databases like them (for the Postgres step later).
RESULTS_FILE = "results.csv"
COLUMNS = ["run_time", "task", "model", "avg_acc", "spread", "runs", "train_s", "guess_s"]


def save_row(row: dict) -> None:
    # Adds one row to the end of results.csv.  row: dict, column name -> value  [{"task": "digits", "avg_acc": 0.97, ...}]
    is_new = not os.path.exists(RESULTS_FILE)              # bool: True the very first time, when the file doesn't exist yet
    with open(RESULTS_FILE, "a", newline="") as f:         # "a" = append: add to the end, never erase what's there
        writer = csv.DictWriter(f, fieldnames=COLUMNS)     # writes dicts as rows, in COLUMNS order
        if is_new:
            writer.writeheader()                           # first time only: the line of column names
        writer.writerow(row)                               # the row itself:  2026-10-23 14:05:09,digits,logistic,0.97,...


def make_models(seed: int, settings: dict) -> list:
    # Returns a list of (name, model) pairs, all fresh and untrained.  [("kNN k=1", KNN object), ...]
    # The neural net uses the seed for its random starting weights;
    # most common, kNN and logistic have no randomness, so they don't take one.
    # settings: the task's knob changes. .get(key, {}) means "the value for key, or an empty dict if it's not there".
    net_knobs = settings.get("neural net", {})       # dict  [fashion: {} -> neural net keeps its defaults]
    logistic_knobs = settings.get("logistic", {})    # dict  [fashion: {"learning_rate": 0.1}]
    # ** unpacks a dict into named inputs:  LogisticRegression(**{"learning_rate": 0.1})
    #   is the same as  LogisticRegression(learning_rate=0.1).   **{} adds nothing.
    return [
        ("most common", MostCommon()),   # dummy: always guesses the most common training answer; every model should beat it
        ("kNN k=1", KNN(k=1)),
        ("kNN k=3", KNN(k=3)),
        ("kNN k=5", KNN(k=5)),
        ("neural net", NeuralNet(seed=seed, **net_knobs)),
        ("logistic", LogisticRegression(**logistic_knobs)),
        # 3 neural nets answering as a team, using the same net knobs. It runs once (no seed), with nets 0, 1, 2.
        ("ensemble x3", Ensemble([NeuralNet(seed=seed * 3 + k, **net_knobs) for k in range(3)])),
    ]


if __name__ == "__main__":     # only runs when you start this file directly, not when another file imports it
    # sys.argv = list[str] of everything typed:  "python compare.py stars"  ->  ["compare.py", "stars"]
    # [1:] drops the file name, leaving the task names:  ["stars"]   (typed nothing extra -> [])
    chosen = sys.argv[1:]
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")   # str: when this run started  ["2026-10-23 14:05:09"]; every row of this run shares it

    # Header row. :<12 means "left-align in a space 12 characters wide" so the columns line up.
    print(f"{'task':<12}{'model':<14}{'avg acc':<10}{'spread':<10}{'runs':<6}{'train s':<10}{'guess s':<10}")

    for task_name, load, settings in TASKS:        # task_name: str ["fashion"], load: function [fashion.load], settings: dict
        # Skip this task if you named tasks and this isn't one of them. An empty list counts as False,
        # so typing no names means nothing gets skipped.
        if chosen and task_name not in chosen:
            continue                               # jump to the next task in TASKS

        train_x, train_y, test_x, test_y = load()  # same data for every model  [stars: (6000, 2000), (6000,), (2000, 2000), (2000,)]

        # enumerate gives (position, item):  i = 0, 1, 2 ...   model_name = "kNN k=1", ...
        # make_models(0, settings) is only used here to get the names; _ means "ignore this part" (the model object).
        for i, (model_name, _) in enumerate(make_models(0, settings)):
            scores = []          # list[float]: accuracy of each run   [digits neural net: 0.963, 0.966, 0.963]
            train_times = []     # list[float]: seconds fit took, each run
            guess_times = []     # list[float]: seconds predict took, each run

            for seed in SEEDS:
                model = make_models(seed, settings)[i][1]    # [i] = the i-th (name, model) pair, [1] = the model part

                start = time.time()
                model.fit(train_x, train_y)        # learn from the training rows
                train_times.append(time.time() - start)

                start = time.time()
                guesses = model.predict(test_x)    # np.ndarray of ints, one guess per test row  [digits: 297 guesses like 7, 3, 3, 5, 3, ...]
                guess_times.append(time.time() - start)

                # guesses == test_y -> array of True/False; mean counts True as 1, False as 0 -> fraction right
                #   guesses [7,3,3] vs answers [7,3,5] -> [True, True, False] -> 2/3 = 0.667
                scores.append(float(np.mean(guesses == test_y)))

                # hasattr(model, "seed") asks: does this model store a seed?  [neural net: True, logistic: False]
                # No seed -> run it once. Either it has no randomness (most common, kNN, logistic), so every run
                # would be identical, or it already averages its own luck (ensemble).
                if not hasattr(model, "seed"):
                    break

            spread = max(scores) - min(scores)     # best run minus worst run: how much luck matters  [0.966 - 0.963 = 0.003]
            # Times are the AVERAGE for one run, not the total.
            print(f"{task_name:<12}{model_name:<14}{np.mean(scores):<10.3f}{spread:<10.3f}{len(scores):<6}"
                  f"{np.mean(train_times):<10.2f}{np.mean(guess_times):<10.2f}")
            save_row({                                     # the same numbers as the printed line, rounded the same way
                "run_time": run_time,
                "task": task_name,
                "model": model_name,
                "avg_acc": round(float(np.mean(scores)), 3),
                "spread": round(spread, 3),
                "runs": len(scores),
                "train_s": round(float(np.mean(train_times)), 2),
                "guess_s": round(float(np.mean(guess_times)), 2),
            })

    print(f"saved to {RESULTS_FILE}")
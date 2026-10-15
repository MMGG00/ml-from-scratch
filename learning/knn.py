import math
from collections import Counter


def distance(point_a: tuple, point_b: tuple) -> float:
    # How different two things are. Each point is a list of measurements,
    # like (sweetness, crunchiness, roundness). Compare them one by one.
    return math.sqrt(sum((number_a - number_b) ** 2 for number_a, number_b in zip(point_a, point_b)))


def k_nearest(new_point: tuple, examples: list, k: int) -> list:
    # examples looks like [((7, 9, 9), "apple"), ((9, 2, 2), "banana")]
    # Sort them by how close they are to the new point, keep the closest k.
    sorted_examples = sorted(examples, key=lambda example: distance(new_point, example[0]))
    return sorted_examples[:k]


def predict(new_point: tuple, examples: list, k: int) -> str:
    # Let the k closest examples vote. Most common label wins.
    nearest = k_nearest(new_point, examples, k)
    labels = [example[1] for example in nearest]
    return Counter(labels).most_common(1)[0][0]

def accuracy(train_examples: list, test_examples: list, k: int) -> float:
    # How often does predict get the right label on examples it hasn't seen?
    correct = 0
    for point, true_label in test_examples:
        guess = predict(point, train_examples, k)
        if guess == true_label:
            correct += 1
    return correct / len(test_examples)


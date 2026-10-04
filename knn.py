import math

def distance(point_a, point_b):
    return math.sqrt(sum((number_a - number_b) ** 2 for number_a, number_b in zip(point_a, point_b)))

def k_nearest(new_point, examples, k):
    sorted_examples = sorted(examples, key=lambda example: distance(new_point, example[0]))
    return sorted_examples[:k]
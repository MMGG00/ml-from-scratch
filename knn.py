import math

def distance(point_a, point_b):
    return math.sqrt(sum((number_a - number_b) ** 2 for number_a, number_b in zip(point_a, point_b)))
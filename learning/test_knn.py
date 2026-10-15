import math

from knn import accuracy, distance, k_nearest, predict

FRUIT = [
    ((7, 9, 9), "apple"),
    ((8, 8, 9), "apple"),
    ((9, 2, 2), "banana"),
    ((8, 3, 1), "banana"),
]


# distance

def test_distance_3_4_5():
    assert distance((0, 0), (3, 4)) == 5.0


def test_distance_same_point_is_zero():
    assert distance((2, 7), (2, 7)) == 0.0


def test_distance_is_symmetric():
    assert distance((1, 2), (4, 6)) == distance((4, 6), (1, 2))


def test_distance_works_with_three_measurements():
    assert math.isclose(distance((0, 0, 0), (1, 2, 2)), 3.0)


# k_nearest

def test_two_closest_are_both_apples_in_order():
    assert k_nearest((7, 8, 8), FRUIT, 2) == [((7, 9, 9), "apple"), ((8, 8, 9), "apple")]


def test_k_larger_than_data_returns_everything():
    assert len(k_nearest((0, 0, 0), FRUIT, 10)) == 4


def test_k_nearest_does_not_change_the_original_list():
    before = list(FRUIT)
    k_nearest((7, 8, 8), FRUIT, 2)
    assert FRUIT == before


# predict

def test_crunchy_round_fruit_is_apple():
    assert predict((7, 8, 8), FRUIT, 3) == "apple"


def test_soft_long_fruit_is_banana():
    assert predict((9, 3, 2), FRUIT, 3) == "banana"


def test_k1_picks_the_single_closest():
    assert predict((8, 3, 2), FRUIT, 1) == "banana"


# accuracy

def test_accuracy_two_of_three():
    test = [((7, 8, 8), "apple"), ((9, 3, 2), "banana"), ((8, 8, 8), "banana")]
    assert math.isclose(accuracy(FRUIT, test, 3), 2 / 3)


def test_accuracy_all_right_is_one():
    test = [((7, 8, 8), "apple"), ((9, 3, 2), "banana")]
    assert accuracy(FRUIT, test, 3) == 1.0


def test_accuracy_all_wrong_is_zero():
    test = [((7, 8, 8), "banana"), ((9, 3, 2), "apple")]
    assert accuracy(FRUIT, test, 3) == 0.0
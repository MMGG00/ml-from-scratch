import math


def squash(number: float) -> float:
    # Turns any number into something between 0 and 1.
    # Big positive -> close to 1. Big negative -> close to 0. Zero -> exactly 0.5.
    return 1 / (1 + math.exp(-number))


def predict_chance(features: list, weights: list, bias: float) -> float:
    # Same weighted sum as before, then squashed into a chance between 0 and 1.
    total = sum(weight * feature for weight, feature in zip(weights, features)) + bias
    return squash(total)


def log_loss(data: list, weights: list, bias: float) -> float:
    # data looks like [([1, 4], 0), ([6, 7], 1)]: (features, answer) where answer is 0 or 1.
    # Being confidently wrong costs a lot. Being confidently right costs almost nothing.
    total = 0
    for features, answer in data:
        chance = predict_chance(features, weights, bias)
        if answer == 1:
            total += -math.log(chance)
        else:
            total += -math.log(1 - chance)
    return total / len(data)


def gradient_descent(data: list, learning_rate: float = 0.1,
                     max_steps: int = 20000, tolerance: float = 1e-10) -> tuple:
    n = len(data)
    number_of_features = len(data[0][0])
    weights = [0] * number_of_features
    bias = 0
    previous_loss = log_loss(data, weights, bias)
    for step in range(1, max_steps + 1):
        weight_slopes = [0] * number_of_features
        bias_slope = 0
        for features, answer in data:
            error = predict_chance(features, weights, bias) - answer
            for i in range(number_of_features):
                weight_slopes[i] += error * features[i] / n
            bias_slope += error / n
        for i in range(number_of_features):
            weights[i] -= learning_rate * weight_slopes[i]
        bias -= learning_rate * bias_slope

        loss = log_loss(data, weights, bias)
        if abs(previous_loss - loss) < tolerance:
            break
        previous_loss = loss
    return weights, bias


def predict_answer(features: list, weights: list, bias: float) -> int:
    # Chance of 50% or more means "yes" (1), otherwise "no" (0).
    return 1 if predict_chance(features, weights, bias) >= 0.5 else 0
def predict_score(features: list, weights: list, bias: float) -> float:
    # features: the measurements, e.g. [hours_studied, hours_slept]
    # weights: one weight per measurement, e.g. [5, 3]
    return sum(weight * feature for weight, feature in zip(weights, features)) + bias


def mean_squared_error(data: list, weights: list, bias: float) -> float:
    # data looks like [([1, 6], 63), ([2, 7], 71)]: (features, real score)
    total = 0
    for features, real_score in data:
        error = predict_score(features, weights, bias) - real_score
        total += error ** 2
    return total / len(data)


def gradient_descent(data: list, learning_rate: float = 0.01,
                     max_steps: int = 50000, tolerance: float = 1e-12) -> tuple:
    # Same idea as before, but now there is one weight (and one slope) per feature.
    n = len(data)
    number_of_features = len(data[0][0])
    weights = [0] * number_of_features
    bias = 0
    previous_error = mean_squared_error(data, weights, bias)
    for step in range(1, max_steps + 1):
        weight_slopes = [0] * number_of_features
        bias_slope = 0
        for features, real_score in data:
            error = predict_score(features, weights, bias) - real_score
            for i in range(number_of_features):
                weight_slopes[i] += 2 * error * features[i] / n
            bias_slope += 2 * error / n
        for i in range(number_of_features):
            weights[i] -= learning_rate * weight_slopes[i]
        bias -= learning_rate * bias_slope

        error = mean_squared_error(data, weights, bias)
        if abs(previous_error - error) < tolerance:
            break
        previous_error = error
    return weights, bias
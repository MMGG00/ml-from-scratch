def predict_line(hours: float, weight: float, bias: float) -> float:
    # The line: score = weight * hours + bias
    return weight*hours + bias


def mean_squared_error(data: list, weight: float, bias: float) -> float:
    # data looks like [(1, 55), (2, 60), (3, 65), (4, 70)], pairs of (hours, real score)
    total = 0
    for hours, real_score in data:
        error = predict_line(hours, weight, bias) - real_score
        total += error ** 2
    return total / len(data)


def gradient_descent(data: list, learning_rate: float = 0.05,
                     max_steps: int = 10000, tolerance: float = 1e-9) -> tuple:
    # Nudge weight and bias downhill until the error stops improving.
    weight = 0
    bias = 0
    n = len(data)
    previous_error = mean_squared_error(data, weight, bias)
    for step in range(1, max_steps + 1):
        weight_slope = 0
        bias_slope = 0
        for hours, real_score in data:
            error = predict_line(hours, weight, bias) - real_score
            weight_slope += 2 * error * hours / n
            bias_slope += 2 * error / n
        weight -= learning_rate * weight_slope
        bias -= learning_rate * bias_slope

        error = mean_squared_error(data, weight, bias)
        if abs(previous_error - error) < tolerance:   # barely improving, so stop
            break
        previous_error = error
    return weight, bias

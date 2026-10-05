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

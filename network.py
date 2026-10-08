import math
import random
from logistic import predict_chance


def make_network(input_size: int, hidden_size: int) -> list:
    # A neuron is [weights, bias]. Every knob starts as a small random number.
    hidden = [[[random.uniform(-1, 1) for _ in range(input_size)], random.uniform(-1, 1)]
              for _ in range(hidden_size)]
    output = [[[random.uniform(-1, 1) for _ in range(hidden_size)], random.uniform(-1, 1)]]
    return [hidden, output]


def forward(inputs: list, layers: list) -> list:
    # Each layer's answers become the next layer's inputs.
    values = inputs
    for layer in layers:
        values = [predict_chance(values, weights, bias) for weights, bias in layer]
    return values


def cost(data: list, layers: list) -> float:
    # Same log loss as logistic.py, using the network's final answer.
    total = 0
    for inputs, answer in data:
        chance = forward(inputs, layers)[0]
        chance = min(max(chance, 1e-12), 1 - 1e-12)  # stops log(0) from crashing
        if answer == 1:
            total += -math.log(chance)
        else:
            total += -math.log(1 - chance)
    return total / len(data)


def train(data: list, layers: list, learning_rate: float = 2.0,
          steps: int = 20000, nudge: float = 1e-5) -> list:
    for step in range(steps):
        base = cost(data, layers)
        if step % 2000 == 0:
            print("step", step, "cost", round(base, 4))

        # 1) Measure: wiggle each knob, see how the cost changes, put it back.
        slopes = []
        for layer in layers:
            for neuron in layer:
                weights = neuron[0]
                for i in range(len(weights)):
                    weights[i] += nudge
                    slopes.append((cost(data, layers) - base) / nudge)
                    weights[i] -= nudge
                neuron[1] += nudge
                slopes.append((cost(data, layers) - base) / nudge)
                neuron[1] -= nudge

        # 2) Move: every knob steps downhill, in the same order we measured.
        k = 0
        for layer in layers:
            for neuron in layer:
                weights = neuron[0]
                for i in range(len(weights)):
                    weights[i] -= learning_rate * slopes[k]
                    k += 1
                neuron[1] -= learning_rate * slopes[k]
                k += 1
    print("final cost", round(cost(data, layers), 4))
    return layers

def backprop(inputs: list, answer: int, layers: list) -> list:
    # Slopes for ONE student, in the same shape as layers.
    hidden, output = layers
    out_weights, out_bias = output[0]

    # Forward, but keep the hidden answers (h) because we need them going back.
    h = [predict_chance(inputs, weights, bias) for weights, bias in hidden]
    chance = predict_chance(h, out_weights, out_bias)

    # Backward. Link 1: how wrong was the answer?
    error = chance - answer

    # Output neuron: same formula as logistic.py, its inputs are the h values.
    output_slopes = [[[error * h_j for h_j in h], error]]

    # Hidden neurons: pass the blame back through the gears.
    hidden_slopes = []
    for j in range(len(hidden)):
        # how wrong  x  output weight  x  squash steepness
        blame = error * out_weights[j] * h[j] * (1 - h[j])
        # x the input, for each weight. The bias has no input, so it just gets the blame.
        hidden_slopes.append([[blame * x for x in inputs], blame])

    return [hidden_slopes, output_slopes]


def train_fast(data: list, layers: list, learning_rate: float = 2.0,
               steps: int = 20000) -> list:
    n = len(data)
    for step in range(steps):
        if step % 2000 == 0:
            print("step", step, "cost", round(cost(data, layers), 4))

        # 1) Measure: backprop every student (nothing moves yet).
        all_slopes = [backprop(inputs, answer, layers) for inputs, answer in data]

        # 2) Move: each knob steps downhill by its AVERAGE slope.
        for L in range(len(layers)):
            for j, neuron in enumerate(layers[L]):
                weights = neuron[0]
                for i in range(len(weights)):
                    slope = sum(s[L][j][0][i] for s in all_slopes) / n
                    weights[i] -= learning_rate * slope
                slope = sum(s[L][j][1] for s in all_slopes) / n
                neuron[1] -= learning_rate * slope

    print("final cost", round(cost(data, layers), 4))
    return layers

if __name__ == "__main__":
    # Students: hours slept (shrunk by /10). Did well (1) only for 7-9 hours.
    data = [([hours / 10], 1 if 7 <= hours <= 9 else 0) for hours in range(2, 15)]

    random.seed(1)
    layers = make_network(input_size=1, hidden_size=2)
    train(data, layers)

    for hours in range(2, 15):
        chance = forward([hours / 10], layers)[0]
        print(hours, "hours slept ->", round(chance, 2))
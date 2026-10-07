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


if __name__ == "__main__":
    # Students: hours slept (shrunk by /10). Did well (1) only for 7-9 hours.
    data = [([hours / 10], 1 if 7 <= hours <= 9 else 0) for hours in range(2, 15)]

    random.seed(1)
    layers = make_network(input_size=1, hidden_size=2)
    train(data, layers)

    for hours in range(2, 15):
        chance = forward([hours / 10], layers)[0]
        print(hours, "hours slept ->", round(chance, 2))
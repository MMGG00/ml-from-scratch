import random
from sklearn.datasets import load_digits
from knn import accuracy

digits = load_digits()

# One example = (the 64 pixel numbers, the digit as text), same shape as the fruit.
examples = [(tuple(pixels), str(label))
            for pixels, label in zip(digits.data.tolist(), digits.target.tolist())]

random.seed(0)            # same shuffle every run, so results can be repeated
random.shuffle(examples)  # mix them so both piles get all kinds of digits

train_examples = examples[:1500]   # the known pictures predict looks at
test_examples = examples[1500:]    # the quiz: we hide their labels

for k in (1, 3, 5, 7):
    score = accuracy(train_examples, test_examples, k)
    print("k =", k, "accuracy =", score)

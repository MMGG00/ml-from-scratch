# ml-from-scratch

Machine learning algorithms written in plain Python, with no ML libraries. The goal is to understand how they work by building them myself.

## What's in here so far

**k-nearest neighbors (kNN)** in `knn.py`. kNN labels a new thing by finding the known examples most like it and letting them vote.

Say you describe fruit with three ratings from 1 to 10: sweetness, crunchiness, roundness. If you know a few apples and bananas, you can label a new fruit by finding the known fruits closest to it and taking the most common label.

| Function | What it does |
|---|---|
| `distance(point_a, point_b)` | How different two points are (Euclidean distance). Works for any number of measurements. |
| `k_nearest(new_point, examples, k)` | The `k` labeled examples closest to the new point. |
| `predict(new_point, examples, k)` | The most common label among those `k` examples. |
| `accuracy(train_examples, test_examples, k)` | The fraction of test examples `predict` gets right. |

## Example

```python
from knn import predict, accuracy

fruit = [
    ((7, 9, 9), "apple"),    # (sweetness, crunchiness, roundness), label
    ((8, 8, 9), "apple"),
    ((9, 2, 2), "banana"),
    ((8, 3, 1), "banana"),
]

print(predict((7, 8, 8), fruit, 3))   # apple
print(predict((9, 3, 2), fruit, 3))   # banana

test = [((7, 8, 8), "apple"), ((9, 3, 2), "banana"), ((8, 8, 8), "banana")]
print(accuracy(fruit, test, 3))       # 0.666..., the last one is mislabeled on purpose
```

## Run the tests

You need Python 3.9 or newer.

```
pip install pytest
python -m pytest
```

On Windows you may need `py -m pytest` instead.

## Known limitations

- **It's slow at prediction time.** Every prediction measures the distance to every training example.
- **Position matters, not just amount.** On pictures, two images of the same digit shifted a few pixels can look far apart.
- **Majority vote is blunt.** With a small `k`, a rare label can get outvoted by a more common one nearby. Weighting closer neighbors more is a possible upgrade.
- **Measurements need similar scales.** If one measurement is in the hundreds and another is 1 to 10, the big one dominates the distance.

## Planned

1. Run kNN on real handwritten digits and measure accuracy for different values of `k`.
2. Linear regression with gradient descent.
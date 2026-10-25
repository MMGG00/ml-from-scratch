import numpy as np

from models.neural_net import softmax

# A small convolutional network for square photos.
# 3x3 filters slide over the photo and make one edge map each, ReLU keeps only the matches,
# pooling keeps the strongest spot in every 2x2 block, and a final layer turns that into chances.
# It learns from small groups of photos at a time (mini-batches) instead of all of them at once.
# Example sizes in [brackets] are for fashion: 28x28 photos, 8 filters, groups of 50, 10 kinds of item.


class CNN:
    def __init__(self, side: int, filters: int = 8, learning_rate: float = 0.1, batch_size: int = 50,
                 max_epochs: int = 30, patience: int = 3, check_fraction: float = 0.2, seed: int = 0):
        assert (side - 2) % 2 == 0, "side - 2 must be even so the edge maps split into 2x2 blocks"
        self.side = side                      # photo width and height in pixels  [28]
        self.filter_count = filters           # how many 3x3 filters to learn  [8]
        self.learning_rate = learning_rate
        self.batch_size = batch_size          # photos per nudge  [50]
        self.max_epochs = max_epochs          # most passes through all the photos
        self.patience = patience              # stop after this many epochs in a row with no better check score
        self.check_fraction = check_fraction
        self.seed = seed

    def patches(self, x: np.ndarray) -> np.ndarray:
        # Every 3x3 square of every photo, laid out as 9 numbers, ready to be matched against the filters.
        photos = x.reshape(-1, self.side, self.side)                                   # [(50, 28, 28)]
        squares = np.lib.stride_tricks.sliding_window_view(photos, (3, 3), axis=(1, 2))  # [(50, 26, 26, 3, 3)]
        return squares.reshape(len(x), self.side - 2, self.side - 2, 9)               # [(50, 26, 26, 9)]

    def forward(self, x: np.ndarray) -> tuple:
        patches = self.patches(x)
        totals = patches @ self.filters + self.filter_biases   # one edge map per filter  [(50, 26, 26, 8)]
        maps = np.maximum(totals, 0)                           # ReLU: negatives become 0
        half = (self.side - 2) // 2                            # [13]
        blocks = maps.reshape(len(x), half, 2, half, 2, self.filter_count)   # cut each map into 2x2 blocks
        pooled = blocks.max(axis=(2, 4))                       # keep the biggest in each block  [(50, 13, 13, 8)]
        flat = pooled.reshape(len(x), -1)                      # one row per photo  [(50, 1352)]
        chances = softmax(flat @ self.output_weights + self.output_biases)   # [(50, 10)]
        return patches, totals, blocks, pooled, flat, chances

    def nudge(self, batch_x: np.ndarray, batch_answers: np.ndarray) -> None:
        # One learning step from one group of photos: guess, measure how wrong, push every number a little.
        patches, totals, blocks, pooled, flat, chances = self.forward(batch_x)
        errors = (chances - batch_answers) / len(batch_x)

        # Final layer: the same slopes as logistic regression.
        output_weight_slopes = flat.T @ errors
        output_bias_slopes = errors.sum(axis=0)

        # Blame flows back to the pooled maps, then only to the spot that won each 2x2 block,
        # then only where ReLU let the number through.
        pooled_blame = (errors @ self.output_weights.T).reshape(pooled.shape)
        winners = blocks == pooled[:, :, None, :, None, :]
        map_blame = (winners * pooled_blame[:, :, None, :, None, :]).reshape(totals.shape)
        total_blame = map_blame * (totals > 0)

        # A filter is used at every spot, so its slope adds up the blame from every spot and every photo.
        filter_slopes = patches.reshape(-1, 9).T @ total_blame.reshape(-1, self.filter_count)
        filter_bias_slopes = total_blame.sum(axis=(0, 1, 2))

        self.output_weights -= self.learning_rate * output_weight_slopes
        self.output_biases -= self.learning_rate * output_bias_slopes
        self.filters -= self.learning_rate * filter_slopes
        self.filter_biases -= self.learning_rate * filter_bias_slopes

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        x = train_x.astype(np.float32)        # float32 is about twice as fast and precise enough here
        n_check = int(len(x) * self.check_fraction)
        check_x, check_y = x[-n_check:], train_y[-n_check:]
        x, y = x[:-n_check], train_y[:-n_check]
        classes = int(train_y.max()) + 1
        answers = np.eye(classes, dtype=np.float32)[y]

        rng = np.random.default_rng(self.seed)
        pooled_size = ((self.side - 2) // 2) ** 2 * self.filter_count     # [13 * 13 * 8 = 1352]
        self.filters = rng.normal(0, 0.3, (9, self.filter_count)).astype(np.float32)
        self.filter_biases = np.zeros(self.filter_count, dtype=np.float32)
        self.output_weights = rng.normal(0, 1 / np.sqrt(pooled_size), (pooled_size, classes)).astype(np.float32)
        self.output_biases = np.zeros(classes, dtype=np.float32)

        best_score = -1.0
        best_knobs = None
        epochs_without_better = 0
        self.history = []
        for epoch in range(self.max_epochs):
            order = rng.permutation(len(x))                  # a new shuffle every epoch
            for start in range(0, len(x), self.batch_size):
                batch = order[start:start + self.batch_size]
                self.nudge(x[batch], answers[batch])

            score = float(np.mean(self.predict(check_x) == check_y))
            self.history.append((epoch, score))
            if score > best_score:
                best_score = score
                self.best_epoch = epoch
                best_knobs = (self.filters.copy(), self.filter_biases.copy(),
                              self.output_weights.copy(), self.output_biases.copy())
                epochs_without_better = 0
            else:
                epochs_without_better += 1
                if epochs_without_better >= self.patience:
                    break

        self.filters, self.filter_biases, self.output_weights, self.output_biases = best_knobs

    def predict_chances(self, test_x: np.ndarray) -> np.ndarray:
        return self.forward(test_x.astype(np.float32))[-1]

    def predict(self, test_x: np.ndarray) -> np.ndarray:
        return self.predict_chances(test_x).argmax(axis=1)
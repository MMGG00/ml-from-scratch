import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from models.torch_net import DEVICE


class TorchCNN:
    # Same network as CNN (3x3 filters -> ReLU -> 2x2 max pooling -> final layer),
    # but PyTorch finds the slopes and does the nudging.
    def __init__(self, side: int, filters: int = 8, learning_rate: float = 0.1, batch_size: int = 50,
                 max_epochs: int = 30, patience: int = 3, check_fraction: float = 0.2, seed: int = 0):
        self.side = side  # photo width and height in pixels  [fashion: 28]
        self.filter_count = filters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.check_fraction = check_fraction
        self.seed = seed
        self.device = DEVICE  # "cuda" or "cpu", so compare.py can show where it ran
        self.model = None
        self.history = []
        self.best_epoch = 0

    def to_photos(self, x: np.ndarray) -> torch.Tensor:
        # Rows of pixels -> a stack of square photos on the device.
        # Conv2d wants (photos, channels, height, width). Our photos are gray, so 1 channel.
        xs = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        return xs.reshape(-1, 1, self.side, self.side)  # [fashion: (rows, 784) -> (rows, 1, 28, 28)]

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        torch.manual_seed(self.seed)

        n_check = int(len(train_x) * self.check_fraction)
        check_x, check_y = train_x[-n_check:], train_y[-n_check:]
        train_x, train_y = train_x[:-n_check], train_y[:-n_check]

        classes = int(train_y.max()) + 1
        half = (self.side - 2) // 2  # side after the filters and pooling  [fashion: 13]

        self.model = nn.Sequential(
            nn.Conv2d(1, self.filter_count, kernel_size=3),  # (rows, 8, 26, 26): one edge map per filter
            nn.ReLU(),
            nn.MaxPool2d(2),  # (rows, 8, 13, 13): biggest number in each 2x2 block
            nn.Flatten(),  # (rows, 1352): each photo's maps in one row
            nn.Linear(self.filter_count * half * half, classes),  # (rows, 10)
        ).to(DEVICE)

        xs = self.to_photos(train_x)
        ys = torch.tensor(train_y, dtype=torch.long, device=DEVICE)
        batches = DataLoader(TensorDataset(xs, ys), batch_size=self.batch_size, shuffle=True)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        best_score = -1.0
        best_knobs = None
        epochs_without_better = 0
        self.history = []

        for epoch in range(self.max_epochs):
            for batch_x, batch_y in batches:
                loss = loss_fn(self.model(batch_x), batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            score = float(np.mean(self.predict(check_x) == check_y))
            self.history.append((epoch, score))
            if score > best_score:
                best_score = score
                self.best_epoch = epoch
                best_knobs = copy.deepcopy(self.model.state_dict())
                epochs_without_better = 0
            else:
                epochs_without_better += 1
                if epochs_without_better >= self.patience:
                    break

        self.model.load_state_dict(best_knobs)

    def predict_chances(self, test_x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            chances = torch.softmax(self.model(self.to_photos(test_x)), dim=1)
        return chances.cpu().numpy()

    def predict(self, test_x: np.ndarray) -> np.ndarray:
        return self.predict_chances(test_x).argmax(axis=1)

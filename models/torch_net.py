import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# Use the graphics card if this computer has one PyTorch can use, otherwise the normal processor.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TorchNet:
    # Same network as NeuralNet (inputs -> 32 hidden with sigmoid -> one score per answer),
    # but PyTorch finds the slopes and does the nudging.
    def __init__(self, hidden_size: int = 32, learning_rate: float = 0.5, batch_size: int = 50,
                 max_epochs: int = 100, patience: int = 10, check_fraction: float = 0.2, seed: int = 0):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.check_fraction = check_fraction
        self.seed = seed
        self.device = DEVICE  # "cuda" or "cpu", so compare.py can show where it ran
        self.model = None  # built in fit, once we know how many features and answers the task has
        self.history = []  # (epoch, check score) after every epoch
        self.best_epoch = 0

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        torch.manual_seed(self.seed)  # same starting weights and shuffles every run with this seed

        # Same check pile split as NeuralNet: the last 20% is only quizzed on, never learned from.
        n_check = int(len(train_x) * self.check_fraction)
        check_x, check_y = train_x[-n_check:], train_y[-n_check:]
        train_x, train_y = train_x[:-n_check], train_y[:-n_check]

        features = train_x.shape[1]  # [digits: 64]
        classes = int(train_y.max()) + 1  # [digits: 10]

        self.model = nn.Sequential(
            nn.Linear(features, self.hidden_size),
            nn.Sigmoid(),
            nn.Linear(self.hidden_size, classes),
        ).to(DEVICE)

        # NumPy arrays become tensors on the device. Labels have to be whole numbers (long) for the loss.
        xs = torch.tensor(train_x, dtype=torch.float32, device=DEVICE)
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

            # After each pass: quiz on the check pile, keep a copy if it's the best so far, stop if stuck.
            score = float(np.mean(self.predict(check_x) == check_y))
            self.history.append((epoch, score))
            if score > best_score:
                best_score = score
                self.best_epoch = epoch
                best_knobs = copy.deepcopy(self.model.state_dict())  # a frozen copy of every weight and bias
                epochs_without_better = 0
            else:
                epochs_without_better += 1
                if epochs_without_better >= self.patience:
                    break

        self.model.load_state_dict(best_knobs)  # go back to the best knobs

    def predict_chances(self, test_x: np.ndarray) -> np.ndarray:
        xs = torch.tensor(test_x, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            chances = torch.softmax(self.model(xs), dim=1)  # (rows, classes), each row adds to 1
        return chances.cpu().numpy()  # back to a NumPy array so compare.py can use it

    def predict(self, test_x: np.ndarray) -> np.ndarray:
        return self.predict_chances(test_x).argmax(axis=1)  # one guess per row  [digits: 297]

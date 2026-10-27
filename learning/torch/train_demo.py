import torch
from torch import nn
from sklearn.datasets import load_digits

torch.manual_seed(0)  # same random starting weights every run

# data: 1797 images of 8x8 pixels, labels 0 to 9
digits = load_digits()
x = torch.tensor(digits.data / 16, dtype=torch.float32)  # (1797, 64), pixels now 0 to 1
y = torch.tensor(digits.target)                          # (1797,) like [0, 1, 2, ...]
train_x, train_y = x[:1500], y[:1500]
test_x, test_y = x[1500:], y[1500:]

# model: 64 pixels -> 32 hidden -> 10 scores, one per digit
model = nn.Sequential(
    nn.Linear(64, 32),
    nn.Sigmoid(),
    nn.Linear(32, 10),
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

for step in range(300):
    scores = model(train_x)          # (1500, 10)
    loss = loss_fn(scores, train_y)  # one number: how wrong, on average

    optimizer.zero_grad()  # clear old slopes
    loss.backward()        # find new slopes
    optimizer.step()       # nudge every weight

    if step % 50 == 0:
        print(step, round(loss.item(), 3))

with torch.no_grad():
    guesses = model(test_x).argmax(dim=1)  # highest score in each row

print("accuracy", round((guesses == test_y).float().mean().item(), 3))
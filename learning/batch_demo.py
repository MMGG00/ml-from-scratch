import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits

torch.manual_seed(0)

digits = load_digits()
x = torch.tensor(digits.data / 16, dtype=torch.float32)
y = torch.tensor(digits.target)
train_x, train_y = x[:1500], y[:1500]
test_x, test_y = x[1500:], y[1500:]

# pairs each photo with its label, then deals them out in shuffled groups of 50
pairs = TensorDataset(train_x, train_y)
batches = DataLoader(pairs, batch_size=50, shuffle=True)

model = nn.Sequential(
    nn.Linear(64, 32),
    nn.Sigmoid(),
    nn.Linear(32, 10),
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

for epoch in range(10):
    for batch_x, batch_y in batches:  # 30 groups of 50, new shuffle each epoch
        loss = loss_fn(model(batch_x), batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(epoch, round(loss.item(), 3))

with torch.no_grad():
    guesses = model(test_x).argmax(dim=1)
print("accuracy", round((guesses == test_y).float().mean().item(), 3))
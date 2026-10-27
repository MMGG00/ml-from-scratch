import torch
from torch import nn

# CrossEntropyLoss gives one number: how wrong the guess was
loss_fn = nn.CrossEntropyLoss()

scores = torch.tensor([[2.0, 1.0, 0.1]])  # 1 row, a raw score for each of 3 classes
right = torch.tensor([0])                  # the right answer is class 0

print(loss_fn(scores, right))

# same thing by hand: softmax, then -log of the right answer's chance
chances = torch.softmax(scores, dim=1)
print(chances)
print(-torch.log(chances[0, 0]))

# if the right answer were class 2 instead, the model is way off
print(loss_fn(scores, torch.tensor([2])))
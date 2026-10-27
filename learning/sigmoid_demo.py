import torch
from torch import nn

# nn.Sigmoid squashes any number into the range 0 to 1
squash = nn.Sigmoid()

scores = torch.tensor([[-2.0, 0.0, 3.0]])  # 1 row, 3 scores

print(squash(scores))
# same math by hand: 1 / (1 + e^-score)
print(1 / (1 + torch.exp(-scores)))
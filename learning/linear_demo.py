import torch
from torch import nn

torch.manual_seed(0)                 # same random start every run, like our seed

layer = nn.Linear(4, 3)              # 4 numbers in, 3 numbers out
print(layer.weight.shape)            # torch.Size([3, 4])
print(layer.bias.shape)              # torch.Size([3])

x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])     # 1 example with 4 numbers: shape (1, 4)
print(layer(x))                               # 3 numbers out
print(x @ layer.weight.T + layer.bias)        # the same 3 numbers, done by hand
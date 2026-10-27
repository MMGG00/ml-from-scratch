import torch

# 1. A tensor is PyTorch's version of a NumPy array.
grid = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(grid.shape)            # torch.Size([2, 2])
print(grid @ grid)           # matrix multiply, same @ as NumPy

# 2. It can live on the GPU. "cuda" means the NVIDIA GPU.
gpu_grid = grid.to("cuda")
print(gpu_grid.device)       # cuda:0  (the first GPU)

# 3. Autograd: PyTorch works out slopes
x = torch.tensor(3.0, requires_grad=True)   # "keep track of how things depend on x"
y = x * x                                   # y = x squared = 9
y.backward()                                # work out the slope of y with respect to x
print(x.grad)                               # tensor(6.)
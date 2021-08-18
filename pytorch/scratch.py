import torch


t = torch.Tensor([range(64) for _ in range(5)])
t = t.reshape((5,4,16))
print(t)

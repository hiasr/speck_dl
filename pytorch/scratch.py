import torch

t = torch.tensor([[[k +16*i for k in range(16)] for i in range(5)] for _ in range(10)])
print(t)
print(t.shape)
t = t.permute((0,2,1))
print(t)
print(t.shape)
t = t.reshape((-1,16,5))
print(t)
print(t.shape)
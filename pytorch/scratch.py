import torch

tensor = torch.tensor([[0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0,
        0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0,
        1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0]], dtype=torch.uint8)
tensor = torch.reshape(tensor, (-1,4,16))
print(tensor[0])
tensor = torch.permute(tensor, (0,2,1))
print(tensor[0])


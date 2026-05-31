import torch

a = torch.tensor([[1,2],[3,4]]).unsqueeze(0)
b = torch.tensor([[5,6],[7,8]]).unsqueeze(0)
c = [a,b]
d = torch.stack(c)
print(d)
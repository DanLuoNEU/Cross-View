import torch


Coeff = torch.randn(1, 161, 50)
dim = Coeff.shape[-1]
Correlation = torch.zeros(1, dim, dim-1)
for j in range(1, dim):
    for i in range(0, dim-1):
        temp1 = Coeff[:, :, i]
        temp2 = Coeff[:, :, j]
        corr = torch.nonzero(temp1)

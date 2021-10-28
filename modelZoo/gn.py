import torch
import time
import torch.nn as nn
norm1 = nn.GroupNorm(8, 512).cuda()
norm2 = nn.GroupNorm(8, 512).cuda()
s = time.time()
for i in range(1000):
    a = torch.zeros((1024, 512)).float().cuda().requires_grad_()
    b = torch.zeros((1024, 512)).float().cuda().requires_grad_()
    ag = norm1(a)
    bg = norm2(b)
    #loss = (ag + bg).reshape(-1).mean()
    loss = ag.mean()
    loss.backward()
    print('i ',i)
    print (ag.grad)
    
print(time.time()-s)
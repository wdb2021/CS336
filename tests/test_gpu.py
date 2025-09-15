import torch; a=torch.rand(1000,1000).cuda(); b=torch.rand(1000,1000).cuda(); c=torch.mm(a,b); print(c.sum())

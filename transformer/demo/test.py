import torch as t
from torch import nn
import torch.functional as F

a = t.arange(0,6)

b = a.view(2,3)

c = b.unsqueeze(1)

print("a:",a)
print("a-size:",a.size())

print("b:",b)
print("b-size:",b.size())

print("c",c)
print("c-size",c.size())

d = b.unsqueeze(0)
print("d:",d)
print("d-size:",d.size())


e = d.squeeze(-3)
print("e:",e)
print("e-size:",e.size())

f = d.squeeze(-2)
print("f",f)
print("f-size:",f.size())









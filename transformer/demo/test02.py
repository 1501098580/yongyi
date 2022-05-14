import torch as t

a = t.arange(0,6)
b = a.view(2,3)

c = b.unsqueeze(1)
d = c.unsqueeze(2)

print("c:",c)
print("c-size:",c.size())

print("*****************************************")
print("d:",d)
print("d-size:",d.size())






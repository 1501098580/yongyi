import torch as t
import numpy as np
import torch

a = np.array([[1,15,50,80,40,30,11,2],
              [1,25,54,44,66,55,2,0],
              [1,4,55,7,8,7,2,0]],dtype=float)


src = t.from_numpy(a)
#print("src:",src)

trg = src

src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
print("src_mask:",src_mask)
print("src_mask-size:",src_mask.size())

trg_len = trg.size(1)
pad_mask = (trg != 0).unsqueeze(1).unsqueeze(2)
# print("pad_mask:",pad_mask)
print("pad_mask-size:",pad_mask.size())

sub_mask = torch.tril(torch.ones(size=(trg_len,trg_len),dtype = torch.uint8)).bool()
# print("sub_mask:",sub_mask)
print("sub-mask-size:",sub_mask.size())

trg_mask = pad_mask & sub_mask
# print("trg_mask:",trg_mask)
print("trg-mask-size:",trg_mask.size())


A_data = np.array([[[2, 3, 5, 8, 12, 1, 1, 5],
                 [6, 1, 1, 2, 3, 1, 2, 4],
                 [12, 3, 4, 5, 6, 7, 8, 9],
                 [2, 3, 5, 8, 12, 1, 1, 5],
                 [6, 1, 1, 2, 3, 1, 2, 4],
                 [12, 3, 4, 5, 6, 7, 8, 9],
                 [6, 1, 1, 2, 3, 1, 2, 4],
                 [12, 3, 4, 5, 6, 7, 8, 9]],

                [[2, 3, 5, 8, 12, 1, 1, 5],
                 [6, 1, 1, 2, 3, 1, 2, 4],
                 [12, 3, 4, 5, 6, 7, 8, 9],
                 [2, 3, 5, 8, 12, 1, 1, 5],
                 [6, 1, 1, 2, 3, 1, 2, 4],
                 [12, 3, 4, 5, 6, 7, 8, 9],
                 [6, 1, 1, 2, 3, 1, 2, 4],
                 [12, 3, 4, 5, 6, 7, 8, 9]],

                [[2, 3, 5, 8, 12, 1, 1, 5],
                 [6, 1, 1, 2, 3, 1, 2, 4],
                 [12, 3, 4, 5, 6, 7, 8, 9],
                 [2, 3, 5, 8, 12, 1, 1, 5],
                 [6, 1, 1, 2, 3, 1, 2, 4],
                 [12, 3, 4, 5, 6, 7, 8, 9],
                 [6, 1, 1, 2, 3, 1, 2, 4],
                 [12, 3, 4, 5, 6, 7, 8, 9]],
                 ],dtype=float)


A_data = torch.from_numpy(A_data)
#print(data)

A_data = A_data.unsqueeze(1)
print("data-size:",A_data.size())
#data-szie [3,1,8,8]  [N, nhead, max_seq_len_q,max_seq_len_k]

energy1 = A_data
energy1 = energy1.masked_fill(src_mask == 0 , 100)
print("src_energy:",energy1)
print("src-energy-size:",energy1.size())

energy2 = A_data
energy2 = energy2.masked_fill(trg_mask == 0 ,100)
print("trg_energy:",energy2)
print("trg-energy-size:",energy2.size())














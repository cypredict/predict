# 一些PyTorch基础练习
# 获得API call sequence邻接矩阵表
import numpy as np
import torch

X = torch.Tensor([[112,274,158,215,274,158,215,298,76,208,76,172,117,172,117,172,76,117,35,60,81,60,81,60,81,60,81,60,81,60,81,60,81,60,81,60,81,60,81,60,81,60,81,60,81,60,81,117,60,81,60,81,208,35,215,35,208,240,117,172,60,81,60,81,225,35,6]])

input_dim_1 = 307
A = torch.zeros((X.size(0), input_dim_1, input_dim_1), dtype = torch.float)
#print(A.size())
#print(x)
#print(X.size(1))
#temp = X[0,:]
#print(temp)
temp = X.numpy()
A_adj = np.zeros([X.size(0), input_dim_1, input_dim_1])

for i in range(temp.shape[0]):
	for j in range(temp.shape[1]):
		x1 = int(temp[i,j])
		if j!=(temp.shape[1]-1):
			x2 = int(temp[i,j+1])
			A_adj[i][x1][x2] = 1
			#print(A_adj[i][x1][x2])
print(A_adj)
#print(sum(sum(A_adj)))

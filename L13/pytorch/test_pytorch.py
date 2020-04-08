# 一些PyTorch基础练习
import numpy as np
import torch

print(torch.eye(3))

points = 100
points = np.arange(points)
print(points)
np.random.shuffle(points)
print(points)
#print()

result = torch.zeros(1, 9517)
print(result)
a = np.ones(3)
x = torch.from_numpy(a)
print(x)

x = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(x)

x = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(x.size())
print(x.view(x.size(0), -1))
# 将tensor的维度换位
#print(x.permute(1, 0))

# 求幂运算
print(x.pow(-1.0))
print(x.pow(2.0))
# 按列求和
print(x.sum(dim = 0))
print(torch.sum(x, 0))
# 按行求和
print(x.sum(dim = 1))
print(torch.sum(x, 1))

# 按行求和，求导数，
temp = x.sum(dim = 1).pow(-1.0)
print(temp)
# 对角矩阵
print(temp.diag_embed())


t = torch.rand(2, 4, 3, 5)
a = np.random.rand(2, 4, 3, 5)
print(t.size())
print(t.stride())
print(t.stride(0))
print(t.stride(1))
print(t.stride(2))
print(t.stride(3))
#print(a.shape)
#print(a)
#print(t.size())
#print(t)

x = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(x.stride())

print(x.narrow(0,0,3))
print(x.narrow(0,1,2))
print(x.narrow(1,1,2))
print(x.narrow(-1,2,1))
x=torch.zeros(3)
print(x)

a=torch.Tensor([[[1,2,3],[4,5,6]]])
b=torch.Tensor([1,2,3,4,5,6])
print(type(a))
print(a.view(1,6))
print(b.view(1,6))
print(a.view(3,2))
# 将Tensor a转换为3行2列
temp = a.view(3,2)
#print(temp.size(-1))
# 数据Tensor temp的大小
print(temp.size())
# 将多行tensor拼接成一行
print(temp.view(1, -1))
# 将多个tensor拼接成一列
print(temp.view(-1, 1))

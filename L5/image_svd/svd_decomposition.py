# 矩阵的奇异值分解
import numpy as np
A = np.array([[1,2],
	        [1,1],
	        [0,0]])

temp1 = np.dot(A, A.T)
temp2 = np.dot(A.T, A)
print(temp1)
print(temp2)

#A = np.array([[5,1],
#	        [1,1]])

lamda1, U1 = np.linalg.eig(temp1)
print(lamda1)
print(U1)
lamda2, U2 = np.linalg.eig(temp2)
print(lamda2)
print(U2)

"""
print('矩阵A: ')
print(A)
print('特征值: ',lamda)
print('特征向量')
print(U)
"""
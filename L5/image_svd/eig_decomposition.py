# 普通矩阵的矩阵分解
import numpy as np
A = np.array([[5,3],
	        [1,1]])

B = np.array([[5,1],
	        [1,1]])

C = np.array([[4,2,-5],
	        [6,4,-9],
	        [5,3,-7]])

def work(A):
	lamda, U = np.linalg.eig(A)
	print('矩阵A: ')
	print(A)
	print('特征值: ',lamda)
	print('特征向量')
	print(U)

work(A)


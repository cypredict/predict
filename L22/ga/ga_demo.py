import numpy as np
import matplotlib.pyplot as plt
from sko.GA import GA

#print(np.random.rand(30))

# 生成数据
x_true = np.linspace(-1.2, 1.2, 30)
# 3阶的y
y_true = x_true ** 3 - x_true + 0.4 * np.random.rand(30)
# 可视化
plt.plot(x_true, y_true, 'o')

# 计算预测结果
def f_fun(x, a, b, c, d):
	return a * x ** 3 + b * x ** 2 + c * x + d

# 计算loss
def obj_fun(p):
	a, b, c, d = p
	# 计算残差 loss
	loss = np.square(f_fun(x_true, a, b, c, d) - y_true).sum()
	return loss

# 使用 scikit-opt 做最优化
ga = GA(func=obj_fun, n_dim=4, size_pop=100, max_iter=500, lb=[-2] * 4, ub=[2] * 4)
best_params, loss = ga.run()
print('best_x:', best_params, '\n', 'best_y:', loss)

# 画出拟合效果图
# 得到预测值
y_predict = f_fun(x_true, *best_params)
fig, ax = plt.subplots()
ax.plot(x_true, y_true, 'o')
ax.plot(x_true, y_predict, '-')
plt.show()

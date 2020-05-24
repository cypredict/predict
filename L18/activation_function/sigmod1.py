import numpy as np
import matplotlib.pyplot as plt

# 绘制sigmoid图
def plot_sigmoid():
    # 设置参数x（起点，终点，间距）
    x = np.arange(-8, 8, 0.2)
    y = sigmoid(x)
    plt.plot(x, y)
    plt.show()

# 绘制sigmoid函数的导数
def plot_ds():
    # 设置参数x（起点，终点，间距）
    x = np.arange(-8, 8, 0.2)
    y = derivative_sigmoid(x)
    plt.plot(x, y)
    plt.show()


# sigmoid函数
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

# sigmoid函数导数
def derivative_sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    ds = s*(1-s)
    return ds

print(sigmoid(4))
#plot_sigmoid()
plot_ds()
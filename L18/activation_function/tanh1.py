import numpy as np
import matplotlib.pyplot as plt

# 绘制tanh图
def plot_tanh():
    # 设置参数x（起点，终点，间距）
    x = np.arange(-8, 8, 0.2)
    y = tanh(x)
    plt.plot(x, y)
    plt.show()

# tanh函数
def tanh(x):
    y=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    return y

print(tanh(4))
plot_tanh()

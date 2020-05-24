import numpy as np
import matplotlib.pyplot as plt

# 绘制relu图
def plot_relu():
    # 设置参数x（起点，终点，间距）
    x = np.arange(-8, 8, 0.2)
    y = relu(x)
    plt.plot(x, y)
    plt.show()

# relu函数
def relu(x):
    y = np.where(x<0,0,x)
    return y

print(relu(4))
plot_relu()

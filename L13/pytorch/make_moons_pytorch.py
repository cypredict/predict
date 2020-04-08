# 使用PyTorch 对随机生成的200个二分类样本进行学习和预测
import sklearn.datasets
import torch
import numpy as np
import matplotlib.pyplot as plt

# 使用make_moon内置生成模型，随机产生二分类数据，200个样本
np.random.seed(33)
X, y = sklearn.datasets.make_moons(200,noise=0.2)
print(X) 
print(y) 
cm = plt.cm.get_cmap('RdYlBu')
plt.scatter(X[:,0],X[:,1],s=40,c=y,cmap=cm)
plt.show()

X = torch.from_numpy(X).type(torch.FloatTensor)
y = torch.from_numpy(y).type(torch.LongTensor)
print(X)
print(y)

import torch.nn as nn
import torch.nn.functional as F
 
# 定义的网络模型继承了nn.Module
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # 定义第1个FC层，使用Linear transformation
        self.fc1 = nn.Linear(2,3)
        # 定义第2个FC层，使用Linear transformation
        self.fc2 = nn.Linear(3,2)
        
    def forward(self,x):
        # 第一层的输出
        x = self.fc1(x)
        # 激活层
        x = F.tanh(x)
        # 输出层
        x = self.fc2(x)
        return x
        
    # 得到预测分类结果 0或者1
    def predict(self,x):
        # 对于输出结果进行softmax计算
        pred = F.softmax(self.forward(x))
        ans = []
        for t in pred:
            if t[0]>t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)
        
 
# 初始化模型
model = Net()
# 定义评估标准（损失函数）
criterion = nn.CrossEntropyLoss()
# 定义优化器
#print('parameters=\n', model.parameters())
print(f'\nParameters: {np.sum([param.numel() for param in model.parameters()])}')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 迭代次数epochs
epochs = 1000
# 存储每次迭代的loss
losses = []
for i in range(epochs):
    # 对输入的X进行预测
    y_pred = model.forward(X)
    # 得到损失loss
    loss = criterion(y_pred,y)
    losses.append(loss.item())
    # 清空之前的梯度
    optimizer.zero_grad()
    # 计算梯度
    loss.backward()
    # 调整权重
    optimizer.step()
   
from sklearn.metrics import accuracy_score
print(model.predict(X))
print(accuracy_score(model.predict(X),y))    
 
# 进行预测，并转换为numpy类型
def predict(x):
    x = torch.from_numpy(x).type(torch.FloatTensor)
    ans = model.predict(x)
    return ans.numpy()
    
# 绘制二分类决策面
def plot_decision_boundary(pred_func,X,y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # 计算决策面
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # np.c_按行连接两个矩阵，就是把两矩阵左右相加
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # 绘制分类决策面
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    # 绘制样本点
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.binary)
    plt.show()
    
plot_decision_boundary(lambda x : predict(x) ,X.numpy(), y.numpy())
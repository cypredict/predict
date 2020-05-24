# 使用RNN处理时序问题
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd

# 数据加载
data = pd.read_csv('./flights.csv')
dataset_ori = data['passengers'].values.astype('float32')
plt.plot(dataset_ori)
plt.show()

#首先我们进行预处理，将数据中 na 的数据去掉，然后将数据标准化到 0 ~ 1 之间。
max_value = np.max(dataset_ori)
min_value = np.min(dataset_ori)
scalar = max_value - min_value
dataset = list(map(lambda x: x / scalar, dataset_ori))
 
'''
look_back: 过去多少个月的乘客数
dataX: 生成的数据集X
dataY: 下一个月的乘客数
'''
def create_dataset(dataset, look_back=2):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)
    
# 创建输入输出
data_X, data_Y = create_dataset(dataset)
#print('data_x=', data_X)
#print('data_y=', data_Y)

# 划分训练集和测试集，70% 作为训练集
train_size = int(len(data_X) * 0.7)
train_X = data_X[:train_size]
train_Y = data_Y[:train_size]
print(train_Y)
'''
    改变数据的维度
    RNN读入的数据维度是 (batch, seq, feature)
    只有一个序列，所以 seq 是 1
    feature 代表依据的几个月份，这里定的是两个月份，所以 feature 就是 2
'''
import torch
print(train_X.shape)
print(train_Y.shape)
train_X = train_X.reshape(-1, 1, 2) #batch, seq, feature
train_Y = train_Y.reshape(-1, 1, 1) #batch, seq, feature
print(train_X.shape)
print(train_Y.shape)
train_x = torch.from_numpy(train_X)
train_y = torch.from_numpy(train_Y)

 
#定义模型
from torch import nn
class rnn_model(nn.Module):
     def __init__(self):
          super(rnn_model, self).__init__()
          input_size, hidden_size, output_size=2, 4, 1
          num_layers=1
          self.rnn = nn.RNN(input_size, hidden_size, num_layers)
          #self.rnn = nn.LSTM(input_size, hidden_size, num_layers) # LSTM参数更多，训练的epoach可能要更多
          #self.rnn = nn.GRU(input_size, hidden_size, num_layers)
          self.out = nn.Linear(hidden_size, output_size)
          
     def forward(self, x):
          x, _ = self.rnn(x) 
          b, s, h = x.shape  #(batch, seq, hidden)
          x = x.view(b*s, h) #转化为线性层的输入方式
          x = self.out(x)
          x = x.view(b, s, -1) #(99, 1, 1)
          return x
# 设置使用GPU
cuda = torch.device('cuda')

#定义好网络结构，输入的维度是 2，因为我们使用两个月的流量作为输入，隐藏层的维度可以任意指定，这里我们选的 4
model = rnn_model()
model = model.cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
 
#开始训练
for epoch in range(1000):
     var_x = train_x.cuda()
     var_y = train_y.cuda()
     # 前向传播
     out = model(var_x)
     # 计算损失函数
     loss = criterion(out, var_y)
     # 反向传播
     optimizer.zero_grad()
     loss.backward()
     optimizer.step()
     if (epoch+1)%100==0:
          print('Epoch:{}, Loss:{:.5f}'.format(epoch+1, loss.item()))
     
# 训练完成之后，我们可以用训练好的模型去预测后面的结果
# eval()会自动把BN和DropOut固定住，不会取平均，而是用训练好的值
model = model.eval()
# 使用全量数据
data_X = data_X.reshape(-1, 1, 2)
data_X = torch.from_numpy(data_X)
pred_test = model(data_X.cuda()) # 测试集的预测结果
 
# 改变输出的格式
pred_test = pred_test.view(-1).data.cpu().numpy()

# 数据反变换
pred_test = list(map(lambda x: x * scalar, pred_test))

# 画出实际结果和预测的结果
plt.plot(pred_test, 'r', label='prediction')
plt.plot(dataset_ori, 'b', label='real')
plt.legend(loc='best')

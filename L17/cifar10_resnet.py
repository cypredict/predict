import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
%matplotlib inline

# 超参数定义
EPOCH = 10              # 训练epoch次数
BATCH_SIZE = 64         # 批训练的数量
LR = 0.001              # 学习率
DOWNLOAD_MNIST = False  # 设置True 可以自动下载数据


# MNIST数据集下载
train_data = datasets.CIFAR10(root='/nas/cifar10/',
                         train=True,                         # 这里是训练集
                         transform=transforms.ToTensor(),    # 将PIL Image或者numpy.ndarray转化为torch.FloatTensor，shape为(C,H,W)，并且归一化到[0.0, 1.0]
                         download=True
                        )

test_data = datasets.CIFAR10(root='/nas/cifar10/',
                        train=False,                         # 测试集
                        transform=transforms.ToTensor(),
                        download=True
                        )

# 输出图像
temp = train_data[1][0].numpy()
print(temp.shape)
temp = temp.transpose(1, 2, 0)
print(temp.shape)
plt.imshow(temp)
plt.show()

# 使用DataLoader进行分批
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

# ResNet Model
model = torchvision.models.resnet50(pretrained=False)
#model = torchvision.models.densenet161(pretrained=False)


#损失函数:这里用交叉熵
criterion = nn.CrossEntropyLoss()
#优化器 这里用SGD
optimizer = optim.Adam(model.parameters(), lr=1e-3)

#device : GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 训练
for epoch in range(EPOCH):
    start_time = time.time()
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # 前向传播
        outputs = model(inputs)
        # 计算损失函数
        loss = criterion(outputs, labels)
        # 清空上一轮梯度
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 参数更新
        optimizer.step()
    print('epoch{} loss:{:.4f} time:{:.4f}'.format(epoch+1, loss.item(), time.time()-start_time))

#保存训练模型
file_name = 'cifar10_resnet.pt'
torch.save(model, file_name)
print(file_name+' saved')


# 测试
model = torch.load(file_name)
model.eval()
correct, total = 0, 0

for data in test_loader:
    images, labels = data
    images, labels = images.to(device), labels.to(device)
    # 前向传播
    out = model(images)
    _, predicted = torch.max(out.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

#输出识别准确率
print('10000测试图像 准确率:{:.4f}%'.format(100.0 * correct / total)) 

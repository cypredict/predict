import torch
from torch import nn, optim
import torchvision
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1), # in_channels, out_channels, kernel_size, stride, padding
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.ReLU(),            
        # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增加输出通道数
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
        # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步 增加了输出通道数。
        # 前两个卷积层后不使用池化层来减小输入的⾼和宽 
            nn.Conv2d(64, 128, 3, 1, 1), 
            nn.Conv2d(128, 256, 3, 1, 1), 
            nn.Conv2d(256, 256, 3, 1, 1), 
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )
        # 这里全连接层的输出个数 ALexNet中的数倍。使用dropout来缓解过拟合。
        self.fc = nn.Sequential(
        # 输出层，10分类
            nn.Linear(256*3*3, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 10),
        )
    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
%matplotlib inline

# 超参数定义
EPOCH = 10               # 训练epoch次数
BATCH_SIZE = 64         # 批训练的数量
LR = 0.001              # 学习率
DOWNLOAD_MNIST = False  # 设置True 可以自动下载数据

# MNIST数据集下载
train_data = datasets.MNIST(root='/nas/mnist/',
                         train=True,                         # 这里是训练集
                         # 数据变换(0, 255) -> (0, 1)
                         transform=transforms.ToTensor(),    # 将PIL Image或者numpy.ndarray转化为torch.FloatTensor，shape为(C,H,W)，并且归一化到[0.0, 1.0]
                        )

test_data = datasets.MNIST(root='/nas/mnist/',
                        train=False,                         # 测试集
                        transform=transforms.ToTensor()
                        )


test_x = test_data.test_data.type(torch.FloatTensor)[:2000]/255.   # shape (2000, 28, 28) value in range(0,1)
test_y = test_data.test_labels.numpy()[:2000]    # 测试集的y，即label

# plot其中一张手写数字图片
print('训练集大小：', train_data.train_data.size())     # 查看训练集数据大小，60000张28*28的图片 (60000, 28, 28)
print('训练集标签个数：', train_data.train_labels.size())   # 查看训练集标签大小，60000个标签 (60000)
plt.imshow(train_data.train_data[0].numpy(), cmap='gray') # plot 训练集第一张图片
plt.title('%i' % train_data.train_labels[0])              # 图片名称，显示真实标签，%i %d十进制整数，有区别，深入请查阅资料
plt.show()                                                # show
# 使用DataLoader进行分批
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

# AlexNet Model
model = AlexNet()
#损失函数:这里用交叉熵
criterion = nn.CrossEntropyLoss()
#优化器 这里用SGD
optimizer = optim.Adam(model.parameters(), lr=1e-3)

#device : GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 训练
for epoch in range(EPOCH):
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

    print('epoch{} loss:{:.4f}'.format(epoch+1, loss.item()))

print("Finished Traning")


#保存训练模型
torch.save(model, 'mnist_alexnet.pt')
model = torch.load('mnist_alexnet.pt')

# 测试
model.eval()
correct = 0
total = 0

for data in test_loader:
    images, labels = data
    images, labels = images.to(device), labels.to(device)
    # 前向传播
    out = model(images)
    _, predicted = torch.max(out.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

#输出识别准确率
print('10000测试图像 准确率:{:.4f}%'.format(100 * correct / total)) 

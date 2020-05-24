# PyTorch分批数据
import torch
import torch.utils.data as Data #将数据分批次需要用到
    
# 设置随机数种子
torch.manual_seed(33)
#设置批次大小
BATCH_SIZE = 8 
 
x = torch.linspace(1, 16, 16)       # 1到16共16个点
y = torch.linspace(16, 1, 16)       # 65到1共16个点
 
print(x)
print(y)

#将x,y读取，转换成Tensor格式
torch_dataset = Data.TensorDataset(x, y) 
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # 最新批数据
    shuffle=True,               # 是否随机打乱数据
    num_workers=2,              # 用于加载数据的子进程
)

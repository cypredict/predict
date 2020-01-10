# 使用LR模型对Avazu CTR进行预估
import pandas as pd
import numpy as np
from dummyPy import OneHotEncoder  # 超大规模数据one-hot编码
from sklearn.linear_model import SGDClassifier  # 梯度下降分类
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt 
import pickle

##==================== 设置文件路径File-Path (fp) ====================##
file_path = '/home/admin/jupyter/avazu/'
train_file = file_path + "train_sample.csv"
test_file  = file_path + "test_sample.csv"

# one-hot编码保存
onehot_encoder_file = file_path + "onehot_encoder"
# LR模型保存
lr_model_file = file_path + "lr/lr_model"
# submission文件保存
submission_file = file_path + "lr/LR_submission.csv"

##==================== LR 训练 ====================##
print(onehot_encoder_file)
onehot_encoder = pickle.load(open(onehot_encoder_file, 'rb'))

# 一个chunk块为5万行
chunksize = 50000
df_train = pd.read_csv(train_file, dtype={'id':str}, index_col=None, chunksize=chunksize, iterator=True)

# 使用LogLoss作为LR的损失函数
lr_model = SGDClassifier(loss='log')  
scores = []

# 使用k和i调整训练规模，训练样本 = 所有样本 / k
k = 100  
i = 1
for chunk in df_train:
    # 根据K drop掉样本
    if i < k: 
        i += 1
        continue
    print('training...')
    i = 1
    df_train_chunk = oh_enc.transform(chunk)
    # LR训练
    feature_train = df_train_chunk.columns.drop(['id', 'click'])
    train_X = df_train_chunk[feature_train]
    train_y = df_train_chunk['click'].astype('int')
    lr_model.partial_fit(train_X, train_y, classes = [0,1])  # fitting
    
    # 训练结果logloss分数
    y_pred = lr_model.predict_proba(train_X)[:, 1]
    score = log_loss(train_y, y_pred)
    scores.append(score)

## 存储训练好的LR模型
pickle.dump(lr_model, open(lr_model_file, 'wb'))
print('LR model saved')

print(scores)
#scores = [0.1, 0.2, 0.3, 0.4]
## 绘制LR训练分数曲线
#f1 = plt.figure(1)
plt.title("LR Training Curve") 
plt.plot(scores)
plt.xlabel('iterations')
plt.ylabel('log_loss')
plt.title('log_loss of training')
plt.grid()
plt.show()

import gc
del lr_model
del df_train
gc.collect()

#数据预处理
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import LabelEncoder
from dummyPy import OneHotEncoder
import random
import pickle  # 存储临时变量

## 读文件
file_path = 'avazu/'
train_file = file_path + 'train_sample.csv'
test_file  = file_path + 'test_sample.csv'

## 下采样写文件
#fp_sub_train_f = file_path + 'sub_train_f.csv'
col_counts_file = file_path + 'col_counts'

## data after selecting features (LR_fun needed)
## and setting rare categories' value to 'other' (feature filtering)
#fp_train_f = file_path + 'train_f.csv'
#fp_test_f  = file_path + 'test_f.csv'

## 存储标签编码和one-hot编码
label_encoder_file = file_path + 'label_encoder'
onehot_encoder_file = file_path + 'onehot_encoder'

##==================== 数据预处理 ====================##
# 包括id, click的feature
cols = ['hour', 'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
# 非features字段
cols_train = ['id', 'click']
cols_test  = ['id']
cols_train.extend(cols)
cols_test.extend(cols)

## 数据加载
print('loading data...')
# 只读10行
df_train_sample = pd.read_csv(train_file, nrows = 10)
#print(df_train_sample)

# 可以分块处理文件
df_train_org = pd.read_csv(train_file, chunksize = 10000000, iterator = True)
df_test_org  = pd.read_csv(test_file,  chunksize = 10000000, iterator = True)

#----- 统计字典 不同值个数 -----#
##  统计sample中的字段 不同值个数
cols_counts = {} 
for col in cols:
    cols_counts[col] = df_train_sample[col].value_counts()
#print(cols_counts)
## 统计训练集中的字段 不同值个数
for chunk in df_train_org:
    for col in cols:
        cols_counts[col] = cols_counts[col].append(chunk[col].value_counts())
## 统计测试集中的字段 不同值个数
for chunk in df_test_org:
    for col in cols:
        cols_counts[col] = cols_counts[col].append(chunk[col].value_counts())
## 统计
for col in cols:
    cols_counts[col] = cols_counts[col].groupby(cols_counts[col].index).sum()
    # 排序
    cols_counts[col] = cols_counts[col].sort_values(ascending=False)   
## 存储value_counts
pickle.dump(cols_counts, open(col_counts_file, 'wb'))
#print(cols_counts)


## 绘制分布
fig = plt.figure(1)
for i, col in enumerate(cols):
    # 一共22个feature，放到11*2个figure里
    ax = fig.add_subplot(11, 2, i+1)
    ax.fill_between(np.arange(len(cols_counts[col])), cols_counts[col].get_values())
    # ax.set_title(col)
plt.show()

## 每个字段，只保存前k个字段值
k = 100
col_index = {}
for col in cols:
    col_index[col] = cols_counts[col][0: k-1].index
    #print(col, col_index[col])

## 对分类变量进行标签编码
lb_enc = {}
for col in cols:
    # 超过前100个value，设置为other
    col_index[col] = np.append(col_index[col], 'other')

for col in cols:
    lb_enc[col] = LabelEncoder()
    lb_enc[col].fit(col_index[col])
    
## 存储标签编码
pickle.dump(lb_enc, open(label_encoder_file, 'wb'))
print(label_encoder_file + ' saved')
## one-hot编码
oh_enc = OneHotEncoder(cols)
#df_train_f = pd.read_csv(fp_train_f, index_col=None, chunksize=5000000, iterator=True)
#df_test_f  = pd.read_csv(fp_test_f, index_col=None, chunksize=5000000, iterator=True)

for chunk in df_train_org:
    oh_enc.fit(chunk)
for chunk in df_test_org:
    oh_enc.fit(chunk)

## 存储one-hot编码
pickle.dump(oh_enc, open(onehot_encoder_file, 'wb'))
print(onehot_encoder_file + ' saved')


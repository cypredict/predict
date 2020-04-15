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
file_path = '/home/admin/avazu/'
fp_train = file_path + 'train.csv'
fp_test  = file_path + 'test.csv'

## 下采样写文件
fp_sub_train_f = file_path + 'sub_train_f.csv'
fp_col_counts = file_path + 'col_counts'

## data after selecting features (LR_fun needed)
## and setting rare categories' value to 'other' (feature filtering)
fp_train_f = file_path + 'train_f.csv'
fp_test_f  = file_path + 'test_f.csv'

## 存储标签编码和one-hot编码
fp_lb_enc = file_path + 'lb_enc'
fp_oh_enc = file_path + 'oh_enc'

##==================== 数据预处理 ====================##
## 特征选择
cols = ['C1', 
        'banner_pos', 
        'site_domain', 
        'site_id',
        'site_category',
        'app_id',
        'app_category', 
        'device_type', 
        'device_conn_type',
        'C14', 
        'C15',
        'C16']

cols_train = ['id', 'click']
cols_test  = ['id']
cols_train.extend(cols)
cols_test.extend(cols)

## 数据加载
print('loading data...')
df_train_ini = pd.read_csv(fp_train, nrows = 10)
df_train_org = pd.read_csv(fp_train, chunksize = 10000000, iterator = True)
df_test_org  = pd.read_csv(fp_test,  chunksize = 10000000, iterator = True)

#----- 统计分类变量 数值个数 -----#
## 初始化
cols_counts = {}  # 统计每个特征的分类数量
for col in cols:
    cols_counts[col] = df_train_ini[col].value_counts()

## 统计训练集
for chunk in df_train_org:
    for col in cols:
        cols_counts[col] = cols_counts[col].append(chunk[col].value_counts())

## 统计测试集
for chunk in df_test_org:
    for col in cols:
        cols_counts[col] = cols_counts[col].append(chunk[col].value_counts())
        
## 统计
for col in cols:
    cols_counts[col] = cols_counts[col].groupby(cols_counts[col].index).sum()
    # sort the counts
    cols_counts[col] = cols_counts[col].sort_values(ascending=False)   

## 存储value_counting
pickle.dump(cols_counts, open(fp_col_counts, 'wb'))

## 绘制分布
fig = plt.figure(1)
for i, col in enumerate(cols):
    ax = fig.add_subplot(4, 3, i+1)
    ax.fill_between(np.arange(len(cols_counts[col])), cols_counts[col].get_values())
    # ax.set_title(col)
plt.show()

## 只保存前K个分类变量
k = 99
col_index = {}
for col in cols:
    col_index[col] = cols_counts[col][0: k].index

df_train_org = pd.read_csv(fp_train, dtype = {'id': str}, chunksize = 10000000, iterator = True)
df_test_org  = pd.read_csv(fp_test,  dtype = {'id': str}, chunksize = 10000000, iterator = True)

## 训练集
hd_flag = True  # add column names at 1-st row
for chunk in df_train_org:
    df = chunk.copy()
    for col in cols:
        df[col] = df[col].astype('object')
        # assign all the rare variables as 'other'
        df.loc[~df[col].isin(col_index[col]), col] = 'other'
    with open(fp_train_f, 'a') as f:
        df.to_csv(f, columns = cols_train, header = hd_flag, index = False)
    hd_flag = False

## 测试集
hd_flag = True  # 第一个chunk需要有header
for chunk in df_test_org:
    df = chunk.copy()
    for col in cols:
        df[col] = df[col].astype('object')
        # 设置其他不常用变量为other
        df.loc[~df[col].isin(col_index[col]), col] = 'other'
    with open(fp_test_f, 'a') as f:
        df.to_csv(f, columns = cols_test, header = hd_flag, index = False)      
    hd_flag = False    

## 对分类变量进行标签编码
lb_enc = {}
for col in cols:
    col_index[col] = np.append(col_index[col], 'other')

for col in cols:
    lb_enc[col] = LabelEncoder()
    lb_enc[col].fit(col_index[col])
    
## 存储标签编码
pickle.dump(lb_enc, open(fp_lb_enc, 'wb'))

## one-hot编码
oh_enc = OneHotEncoder(cols)

df_train_f = pd.read_csv(fp_train_f, index_col=None, chunksize=5000000, iterator=True)
df_test_f  = pd.read_csv(fp_test_f, index_col=None, chunksize=5000000, iterator=True)

for chunk in df_train_f:
    oh_enc.fit(chunk)
for chunk in df_test_f:
    oh_enc.fit(chunk)
    
## 存储one-hot编码
pickle.dump(oh_enc, open(fp_oh_enc, 'wb'))


# 计算总训练样本 约46M
n = sum(1 for line in open(fp_train_f)) - 1 
# 保存下采样训练样本 2M
s = 2000000

## 设置哪些行不需要读 skip，不需要读的行数为n-s
skip = sorted(random.sample(range(1, n+1), n-s)) 
df_train = pd.read_csv(fp_train_f, skiprows = skip)
df_train.columns = cols_train

## 存储下采样的结果
df_train.to_csv(fp_sub_train_f, index=False) 


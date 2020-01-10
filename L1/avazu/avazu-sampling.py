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
train_file = file_path + 'train.csv'
test_file  = file_path + 'test.csv'
col_counts_file = file_path + 'col_counts'

## 存储标签编码和one-hot编码
fp_lb_enc = file_path + 'lb_enc'
fp_oh_enc = file_path + 'oh_enc'

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

# sample前10000行数据
df_train_org = pd.read_csv(train_file, nrows=10000)
df_test_org  = pd.read_csv(test_file, nrows=10000)
df_train_org.to_csv('train_sample.csv')
df_test_org.to_csv('test_sample.csv')
print('sample saved')


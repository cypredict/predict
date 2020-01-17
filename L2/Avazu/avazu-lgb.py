import pandas as pd
import numpy as np

file_path = '/home/admin/avazu/train.csv'
data = pd.read_csv(file_path,nrows = 300000)
print(data.head())
print(data.columns)
print(data['click'].value_counts())

data = data.drop(['id'], axis=1)
print(pd.factorize(data['site_id'])[0])

for feature in data.drop('click', axis=1).columns:
    data[feature] = pd.factorize(data[feature])[0]
#print(data.head())

# 切分训练集，测试集
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.2)

# 使用LightGBM
import lightgbm as lgb
clf = lgb.LGBMClassifier(is_unbalanced=False, slient=False)
cates = list(train.drop('click', axis=1).columns)
print(cates)

clf.fit(train.drop('click', axis=1), train['click'], categorical_feature=cates, verbose=5)
predict = clf.predict_proba(test.drop('click', axis=1))[:, 1]

# 打印二分类交叉熵
from sklearn.metrics import log_loss
print(log_loss(test['click'], predict))
import pandas as pd
import numpy as np

file_path = '/home/admin/avazu/train.csv'
data = pd.read_csv(file_path, nrows =300000)
print(data.head())
print(data.columns)
print(data['click'].value_counts())

data = data.drop(['id'], axis=1)
pd.factorize(data['site_id'])[0]

for fea in data.drop('click', axis=1).columns:
    data[fea] = pd.factorize(data[fea])[0]

from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.2)

# 使用XGBoost
import xgboost as xgb
param = {'boosting_type':'gbdt',
                         'objective' : 'binary:logistic',
                         #'eval_metric' : 'auc',
                         'eval_metric' : 'logloss',
                         'eta' : 0.01,
                         'max_depth' : 15,
                         'colsample_bytree':0.8,
                         'subsample': 0.9,
                         'subsample_freq': 8,
                         'alpha': 0.6,
                         'lambda': 0,
        }

train_data = xgb.DMatrix(train.drop('click', axis=1), label=train['click'])
test_data = xgb.DMatrix(test.drop('click', axis=1), label=test['click'])
model = xgb.train(param, train_data, evals=[(train_data, 'train'), (test_data, 'valid')], num_boost_round = 500, early_stopping_rounds=50, verbose_eval=25)

cates = list(train.drop('click', axis=1).columns)
print('cates=', cates)

predict = model.predict(test_data)

# 打印二分类交叉熵
from sklearn.metrics import log_loss
print(log_loss(test['click'], predict))
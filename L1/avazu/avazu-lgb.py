import pandas as pd
import numpy as np

file_path = '/home/admin/jupyter/avazu/train_sample.csv'
data = pd.read_csv(file_path,nrows =300000)
print(data.head())
print(data.columns)
print(data['click'].value_counts())

data = data.drop(['id'], axis=1)
pd.factorize(data['site_id'])[0]

for fea in data.drop('click', axis=1).columns:
    data[fea] = pd.factorize(data[fea])[0]

from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.2)

import lightgbm as lgb
clf = lgb.LGBMClassifier(is_unbalanced=False, slient=False)
cal = list(train.drop('click', axis=1).columns)
print(cal)

clf.fit(train.drop('click', axis=1), train['click'], categorical_feature=cal, verbose=5)
predict = clf.predict_proba(test.drop('click', axis=1))[:, 1]

def celoss(target, predict):
    target = np.array(target)
    predict = np.array(predict)
    return -(target * np.log(predict) + (1 - target) * np.log(1 - predict)).mean()

print(celoss(test['click'], predict))


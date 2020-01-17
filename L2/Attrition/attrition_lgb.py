import pandas as pd
train=pd.read_csv('train.csv',index_col=0)
test=pd.read_csv('test.csv',index_col=0)

#print(train['Attrition'].value_counts())
# 处理Attrition字段
train['Attrition']=train['Attrition'].map(lambda x:1 if x=='Yes' else 0)
from sklearn.preprocessing import LabelEncoder
# 查看数据是否有空值
#print(train.isna().sum())

# 去掉没用的列 员工号码，标准工时（=80）
train = train.drop(['EmployeeNumber', 'StandardHours'], axis=1)
test = test.drop(['EmployeeNumber', 'StandardHours'], axis=1)

# 对于分类特征进行特征值编码
attr=['Age','BusinessTravel','Department','Education','EducationField','Gender','JobRole','MaritalStatus','Over18','OverTime']
lbe_list=[]
for feature in attr:
    lbe=LabelEncoder()
    train[feature]=lbe.fit_transform(train[feature])
    test[feature]=lbe.transform(test[feature])
    lbe_list.append(lbe)
#train.to_csv('temp.csv')
#print(train)

import lightgbm as lgb
from sklearn.model_selection import train_test_split
# param = {
#     'num_leaves':41,
#     'boosting_type': 'gbdt',
#     'objective':'binary',
#     'max_depth':15,
#     'learning_rate':0.001,
#     'metric':'binary_logloss'}
param = {'boosting_type':'gbdt',
                         'objective' : 'binary', #
                         #'metric' : 'binary_logloss',
                         'metric' : 'auc',
#                          'metric' : 'self_metric',
                         'learning_rate' : 0.01,
                         'max_depth' : 15,
                         'feature_fraction':0.8,
                         'bagging_fraction': 0.9,
                         'bagging_freq': 8,
                         'lambda_l1': 0.6,
                         'lambda_l2': 0,
#                          'scale_pos_weight':k,
#                         'is_unbalance':True
        }
X_train, X_valid, y_train, y_valid = train_test_split(train.drop('Attrition',axis=1), train['Attrition'], test_size=0.2, random_state=42)
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid)


model = lgb.train(param,train_data,valid_sets=[train_data,valid_data],num_boost_round = 10000 ,early_stopping_rounds=200,verbose_eval=25, categorical_feature=attr)
predict=model.predict(test)
#print(predict)
test['Attrition']=predict
test['Attrition']=test['Attrition'].map(lambda x:1 if x>=0.5 else 0)
#test['Attrition']=predict
test[['Attrition']].to_csv('submit_lgb.csv')

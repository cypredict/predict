import pandas as pd
train=pd.read_csv('train.csv',index_col=0)
test=test1=pd.read_csv('test.csv',index_col=0)

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
#print(train)

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
X_train, X_valid, y_train, y_valid = train_test_split(train.drop('Attrition',axis=1), train['Attrition'], test_size=0.2, random_state=42)


mms = MinMaxScaler(feature_range=(0, 1))
X_train = mms.fit_transform(X_train)
X_valid = mms.fit_transform(X_valid)
test = mms.fit_transform(test)

model = SVC(kernel='rbf', 
			gamma="auto",
			max_iter=1000,
			random_state=33,
			verbose=True,
			tol=1e-5,
			cache_size=50000
		   )
#print(X_train)
#print(y_train)
#print(sum(y_train))

model = LinearSVC(
			max_iter=1000,
			random_state=33,
			verbose=True,
		   )
model.fit(X_train, y_train)
predict = model.predict(test)
print(predict)
#print(test)
#predict = model.predict_proba(test)[:, 1]
test1['Attrition']=predict

# 转化为二分类输出
#test['Attrition']=test['Attrition'].map(lambda x:1 if x>=0.5 else 0)
test1[['Attrition']].to_csv('submit_svc.csv')

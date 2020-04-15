# 使用DCN模型对Avazu CTR进行预估
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from deepctr.models import DCN #使用Deep & Cross
from deepctr.inputs import SparseFeat,get_feature_names
import pickle


##==================== 设置文件路径File-Path (fp) ====================##
file_path = '/home/admin/avazu/'
fp_train_f = file_path + "sub_train_f.csv" #使用小样本进行训练

##==================== DCN 训练 ====================##
data = pd.read_csv(fp_train_f, dtype={'id':str}, index_col=None)
print('data loaded')

#数据加载
sparse_features = ['C1', 'banner_pos', 'site_domain', 'site_id','site_category','app_id','app_category', 'device_type', 'device_conn_type','C14', 'C15','C16']
target = ['click']

# 对特征标签进行编码
for feature in sparse_features:
    lbe = LabelEncoder()
    data[feature] = lbe.fit_transform(data[feature])
# 计算每个特征中的 不同特征值的个数
fixlen_feature_columns = [SparseFeat(feature, data[feature].nunique()) for feature in sparse_features]
linear_feature_columns = fixlen_feature_columns
dnn_feature_columns = fixlen_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
print(fixlen_feature_columns)
print(feature_names)

# 将数据集切分成训练集和测试集
train, test = train_test_split(data, test_size=0.2)
train_model_input = {name:train[name].values for name in feature_names}
test_model_input = {name:test[name].values for name in feature_names}

# 使用DCN进行训练
#model = DCN(linear_feature_columns, dnn_feature_columns, task='regression')
model = DCN(linear_feature_columns, dnn_feature_columns, task='binary')
model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )
history = model.fit(train_model_input, train[target].values, batch_size=256, epochs=1, verbose=True, validation_split=0.2, )
# 使用DCN进行预测
pred_ans = model.predict(test_model_input, batch_size=256)
# 输出RMSE或MSE
mse = round(mean_squared_error(test[target].values, pred_ans), 4)
rmse = mse ** 0.5
print("test RMSE", rmse)

# 输出LogLoss
from sklearn.metrics import log_loss
score = log_loss(test[target].values, pred_ans)
print("LogLoss", score)
#from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import BaselineOnly, KNNBasic, NormalPredictor
from surprise import accuracy
from surprise.model_selection import KFold
#import pandas as pd

# 数据读取
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file('./ratings.csv', reader=reader)
train_set = data.build_full_trainset()

# ALS优化
bsl_options = {'method': 'als','n_epochs': 5,'reg_u': 12,'reg_i': 5}
# SGD优化
#bsl_options = {'method': 'sgd','n_epochs': 5}
algo = BaselineOnly(bsl_options=bsl_options)
#algo = BaselineOnly()
#algo = NormalPredictor()

# 定义K折交叉验证迭代器，K=3
kf = KFold(n_splits=3)
for trainset, testset in kf.split(data):
    # 训练并预测
    algo.fit(trainset)
    predictions = algo.test(testset)
    # 计算RMSE
    accuracy.rmse(predictions, verbose=True)

uid = str(196)
iid = str(302)
# 输出uid对iid的预测结果
pred = algo.predict(uid, iid, r_ui=4, verbose=True)




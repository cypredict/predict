from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
#from surprise import evaluate, print_perf
from surprise import Reader
from surprise import BaselineOnly, KNNBasic, KNNBaseline, SlopeOne
from surprise import accuracy
from surprise.model_selection import KFold
import pandas as pd
import io
import pandas as pd

# 读取物品（电影）名称信息
def read_item_names():
    file_name = ('./movies.csv') 
    data = pd.read_csv('./movies.csv')
    rid_to_name = {}
    name_to_rid = {}
    for i in range(len(data['movieId'])):
        rid_to_name[data['movieId'][i]] = data['title'][i]
        name_to_rid[data['title'][i]] = data['movieId'][i]

    return rid_to_name, name_to_rid 

# 数据读取
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file('./ratings.csv', reader=reader)
train_set = data.build_full_trainset()


# 使用SlopeOne算法
algo = SlopeOne()
algo.fit(train_set)
# 对指定用户和商品进行评分预测
uid = str(196) 
iid = str(302) 
pred = algo.predict(uid, iid, r_ui=4, verbose=True)
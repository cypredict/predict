# 18支亚洲球队聚类
from scipy.cluster.hierarchy import dendrogram, ward
#from sklearn.cluster.AgglomerativeClustering
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 数据加载
data = pd.read_csv('team_cluster_data.csv', encoding='gbk')
train_x = data[["2019国际排名","2018世界杯排名","2015亚洲杯排名"]]
kmeans = KMeans(n_clusters=3)
# 规范化到 [0,1] 空间
min_max_scaler=preprocessing.MinMaxScaler()
train_x=min_max_scaler.fit_transform(train_x)
#print(train_x)
# kmeans 算法
model = AgglomerativeClustering(linkage='ward', n_clusters=3)
y = model.fit_predict(train_x)
print(y)

linkage_matrix = ward(train_x)
dendrogram(linkage_matrix)
plt.show()

from pyspark.ml.recommendation import ALS
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *
import pandas as pd
sc = SparkContext()
sql_sc = SQLContext(sc)

pd_df_ratings = pd.read_csv('./ratings_small.csv')
pyspark_df_ratings = sql_sc.createDataFrame(pd_df_ratings)
pyspark_df_ratings = pyspark_df_ratings.drop('Timestamp')
#print(pyspark_df_ratings.show(5, truncate=False))

# 创建ALS模型
als = ALS(rank=3, maxIter = 10, regParam=0.1, userCol= 'userId', itemCol='movieId', ratingCol='rating')
model = als.fit(pyspark_df_ratings)
# 对userId=100进行Top-N推荐
recommendations = model.recommendForAllUsers(5)
print(recommendations.where(recommendations.userId == 100).collect())

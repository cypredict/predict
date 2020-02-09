# 使用pyspark-ALS进行矩阵分解
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS

print("使用Spark-ALS算法") 
sc = SparkContext('local', 'MovieRec')
# 读取数据，需要第一行不是列名
rawUserData = temp = sc.textFile('./ratings_small_without_header.csv')
print(rawUserData.count())
print(rawUserData.first())

rawRatings = rawUserData.map(lambda line: line.split(",")[:3])
print(rawRatings.take(5))
training_RDD = rawRatings.map(lambda x: (x[0], x[1], x[2]))

# 模型训练
rank = 3
model = ALS.train(training_RDD, rank, seed=5, iterations=10, lambda_=0.1)
# 针对user_id = 100的用户进行Top-N推荐
print(model.recommendProducts(100, 5))


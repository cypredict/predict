# PySpark 酒店推荐系统
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.ml.feature import HashingTF, IDF
#from pyspark.ml.feature import NGram

# 创建SparkSession，2.0版本之后只需要创建一个SparkSession即可
spark=SparkSession \
        .builder \
        .appName('hotel_rec_app') \
        .getOrCreate()

# 从CSV文件中读取
df = spark.read.csv("Seattle_Hotels.csv", header=True, inferSchema=True)
# 数据探索
df.show(20)
print('数据集中的酒店个数：', df.count())

# 将desc进行分词
tokenizer = Tokenizer(inputCol="desc", outputCol="desc_words")
df = tokenizer.transform(df)
df.show()

df.select('desc_words').show()
#print('数据集中的酒店个数：', df.length)

def print_description(index):
    df.where('id='+str(index)).show()
print('第10个酒店的描述：')
print_description(10)

# 停用词
add_stopwords = ['the', 'of', 'in', 'a', 'an', 'at', 'as', 'on', 'for', 'it', 'we', 'you', 'want', 'up', 'to', 'if', 'are', 'is', 'and', 'our', 'with', 'from', '-', 'your', 'so']
stopwords_remover  = StopWordsRemover(inputCol='desc_words', outputCol='desc_words_filtered').setStopWords(add_stopwords)
df = stopwords_remover.transform(df)

# 计算每篇文档的TF-IDF
hashingTF =HashingTF(inputCol='desc_words_filtered', outputCol="desc_words_tf")
tf = hashingTF.transform(df).cache()
idf = IDF(inputCol='desc_words_tf', outputCol="desc_words_tfidf").fit(tf)
tfidf = idf.transform(tf).cache()
print('\n 每个酒店的TFIDF')
tfidf.select('desc_words_tfidf').show(truncate=False)


# 数据规范化
from pyspark.ml.feature import Normalizer
normalizer = Normalizer(inputCol="desc_words_tfidf", outputCol="norm")
tfidf = normalizer.transform(tfidf)
tfidf.select("id", "norm").show()

import pyspark.sql.functions as psf 
from pyspark.sql.types import DoubleType
dot_udf = psf.udf(lambda x,y: float(x.dot(y)), DoubleType())
#tfidf = tfidf.alias("a1").join(tfidf.alias("a2"), psf.col("a1.id") < psf.col("a2.id")).withColumn('similarity', dot_udf("a1.norm", "a2.norm"))
#tfidf.show()

tfidf = tfidf.alias("a1").join(tfidf.alias("a2"), psf.col("a1.id") < psf.col("a2.id"))\
        .select(
            psf.col("a1.name"),
            psf.col("a1.id").alias("id1"), 
            psf.col("a2.id").alias("id2"), 
            dot_udf("a1.norm", "a2.norm").alias("similarity"))\
        .sort("id1", "id2")

tfidf.show(100)

# 基于相似度和指定的酒店name，推荐TOP10酒店
def recommendations(name):
    temp = tfidf.where('name="'+name+'"').sort('similarity', ascending=False).limit(10)
    return temp.select('id2', 'similarity')

rec = recommendations('Hilton Seattle Airport & Conference Center')
rec.show()
rec = recommendations('The Bacon Mansion Bed and Breakfast')
rec.show()


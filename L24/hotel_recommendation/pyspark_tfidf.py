# 使用pyspark计算文档的TFIDF
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF

# 创建SparkSession，2.0版本之后只需要创建一个SparkSession即可
spark=SparkSession \
        .builder \
        .appName('tfidf_app') \
        .getOrCreate()

# 加载数据
documents = spark.createDataFrame([
    (0, "我 非常 喜欢 看 电视剧", "data1"),
    (1, "我 喜欢 看 电视剧", "data2"),
    (2, "我 喜欢 吃 苹果","data3"),
    (3, "我 喜欢 吃 苹果 看 电视剧","data4")], ["id", "doc_text", "other"])

# 转化为视图
documents.registerTempTable("doc_table")
df= spark.sql("SELECT id, doc_text FROM doc_table")
print('df=')
df.show() # 打印前20行
print(df.collect()) # 转化为列表
# id
df.select('id').show() 

# 将desc进行分词
tokenizer = Tokenizer(inputCol="doc_text", outputCol="doc_words")
df = tokenizer.transform(df)
df.show()

# 计算每篇文档的TF-IDF
hashingTF = HashingTF(inputCol='doc_words', outputCol="doc_words_tf")
#hashingTF = HashingTF()
tf = hashingTF.transform(df).cache()
idf = IDF(inputCol='doc_words_tf', outputCol="doc_words_tfidf").fit(tf)
tfidf = idf.transform(tf).cache()
print('\n 每个文档的TFIDF')
tfidf.select('doc_words_tfidf').show(truncate=False)

# 数据规范化，默认为2阶范式
from pyspark.ml.feature import Normalizer
normalizer = Normalizer(inputCol="doc_words_tfidf", outputCol="norm") #默认.setP(2.0)
tfidf = normalizer.transform(tfidf)
tfidf.select('norm').show(truncate=False)

import pyspark.sql.functions as psf 
from pyspark.sql.types import DoubleType
dot_udf = psf.udf(lambda x,y: float(x.dot(y)), DoubleType())
tfidf.alias("a1").join(tfidf.alias("a2"), psf.col("a1.id") < psf.col("a2.id"))\
    .select(
        psf.col("a1.id").alias("id1"), 
        psf.col("a2.id").alias("id2"), 
        dot_udf("a1.norm", "a2.norm").alias("dot"))\
    .sort("id1", "id2")\
    .show()

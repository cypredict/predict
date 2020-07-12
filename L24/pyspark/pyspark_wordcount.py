# Spark+python 进行wordCount
from pyspark.sql import SparkSession
 
spark = SparkSession\
    .builder\
    .appName("PythonWordCount")\
    .getOrCreate()

# 将文件转换为RDD对象
lines = spark.read.text("input.txt").rdd.map(lambda r: r[0])
print(lines)
counts = lines.flatMap(lambda x: x.split(' ')) \
              .map(lambda x: (x, 1)) \
              .reduceByKey(lambda x, y: x + y)
output = counts.collect()
for (word, count) in output:
    print("%s: %i" % (word, count))

spark.stop()

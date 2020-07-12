# pyspark读写dataframe
from pyspark.sql import SparkSession

spark=SparkSession \
        .builder \
        .appName('first_app') \
        .getOrCreate()

# 从集合中创建RDD
rdd = spark.sparkContext.parallelize([
    (1001, "张飞", 8341, "坦克"),
    (1002, "关羽", 7107, "战士"),
    (1003, "刘备", 6900, "战士")
])

# 指定模式, StructField(name,dataType,nullable)
# name: 该字段的名字，dataType：该字段的数据类型，nullable: 指示该字段的值是否为空
from pyspark.sql.types import StructType, StructField, LongType, StringType  # 导入类型
schema = StructType([
    StructField("id", LongType(), True),
    StructField("name", StringType(), True),
    StructField("hp", LongType(), True), #生命值
    StructField("role_main", StringType(), True)
])

# 对RDD应用该模式并且创建DataFrame
heros = spark.createDataFrame(rdd, schema)
heros.show()

# 利用DataFrame创建一个临时视图
heros.registerTempTable("HeroGames")
# 查看DataFrame的行数
print(heros.count())

# 使用自动类型推断的方式创建dataframe
data = [(1001, "张飞", 8341, "坦克"),
        (1002, "关羽", 7107, "战士"),
        (1003, "刘备", 6900, "战士")]
df = spark.createDataFrame(data, schema=['id', 'name', 'hp', 'role_main'])

print(df) #只能显示出来是DataFrame的结果
df.show() #需要通过show将内容打印出来
print(df.count())


# 从CSV文件中读取
heros = spark.read.csv("./heros.csv", header=True, inferSchema=True)
heros.show()

# 需要将mysql-jar驱动放到spark\jars下面
# 驱动下载：https://www.mysql.com/products/connector/
df = spark.read.format('jdbc').options(
    url='jdbc:mysql://localhost:3306/wucai?useUnicode=true&useJDBCCompliantTimezoneShift=true&useLegacyDatetimeCode=false&serverTimezone=Asia/Shanghai',
    #driver="com.mysql.jdbc.Driver",
    dbtable='heros',
    user='root',
    password='passw0rdcc4'
    ).load()
print('连接JDBC，调用Heros数据表')
df.show()

# 传入SQL语句
sql="(SELECT id, name, hp_max, role_main FROM heros WHERE role_main='战士') t"
df = spark.read.format('jdbc').options(
    url='jdbc:mysql://localhost:3306/wucai?useUnicode=true&useJDBCCompliantTimezoneShift=true&useLegacyDatetimeCode=false&serverTimezone=Asia/Shanghai',
    dbtable=sql,
    user='root',
    password='passw0rdcc4' 
    ).load()
df.show()
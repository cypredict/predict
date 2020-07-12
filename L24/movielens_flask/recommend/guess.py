# coding=utf-8
from spark import RecommendationEngine
from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession

TOTAL_USERS = 0
TOTAL_MOVIES = 0
RECOMMEND_NUMS = 5
DEBUG = True
def run_recommond(ID):
    conf = SparkConf().setAppName("recommond")
    sc = SparkContext(conf=conf)
    spark_session = SparkSession.builder.config(conf=conf).getOrCreate()
    #dataset_path = r'D:\app\python\RedDragon\recommend\dataset\data_model'
    dataset_path = r'./recommend/dataset/data_model'
    # 调用编写好的推荐引擎
    recommendation = RecommendationEngine(sc, dataset_path, spark_session)
    # 给用户推荐Top10电影
    top_ratings = recommendation.get_top_ratings(user_id=ID, movies_count=10)
    # 推荐完成后，sc停止
    try:
        sc.stop()
    except:
        pass
    return top_ratings


#print run_recommond(33)
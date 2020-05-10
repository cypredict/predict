# sklearn, TF-IDF使用
import jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

# 加载数据
wordslist = ["我非常喜欢看电视剧","我非常喜欢旅行","我非常喜欢吃苹果", '我非常喜欢跑步'] 
# 分词
textTest = [' '.join(jieba.cut(words)) for words in wordslist]

""" 统计词频 """
vectorizer = CountVectorizer()
count = vectorizer.fit_transform(textTest)
# 特征
print(vectorizer.get_feature_names())
# 单词与词频统计
print(vectorizer.vocabulary_)
# 词频
#print('count=', count)
print(count.toarray()) # 导出的是Object类型数组

""" 计算tfidf """
# TfidfTransformer是统计CountVectorizer中每个词语的tf-idf权值
transformer = TfidfTransformer()
# TFIDF矩阵
tfidf_matrix = transformer.fit_transform(count)
print(tfidf_matrix.toarray())


# TfidfVectorizer可以把CountVectorizer, TfidfTransformer合并起来，直接生成tfidf值
tfidf_vec = TfidfVectorizer() 
tfidf_matrix = tfidf_vec.fit_transform(textTest)
print(tfidf_vec.get_feature_names())
print(tfidf_vec.vocabulary_)
print(tfidf_matrix.toarray())

""" 单词与单词之间的余弦相似度 """
from sklearn.metrics.pairwise import cosine_similarity
print('cosine_similarity=\n', cosine_similarity(tfidf_matrix, tfidf_matrix))

# 测试集
test = [' '.join(jieba.cut("我喜欢看电视剧"))]
# 得到tfidf vec
test_vec = tfidf_vec.transform(test)
print('cosine_similarity=\n', cosine_similarity(test_vec, tfidf_matrix))

# Gensim, TF-IDF使用
import jieba
from gensim import corpora, models, similarities

# 加载数据
wordslist = ["我非常喜欢看电视剧","我非常喜欢旅行","我非常喜欢吃苹果", '我非常喜欢跑步'] 
# 分词
textTest = [[word for word in jieba.cut(words)] for words in wordslist]
print('分词：', textTest)
# 生成字典
dictionary = corpora.Dictionary(textTest)
print('字典：', dictionary)
print('单词词频:', dictionary.dfs)  # 字典词频，{单词id，在多少文档中出现}
print('文档数目:', dictionary.num_docs)  # 文档数目
print('所有词的个数:', dictionary.num_pos)  # 所有词的个数
featurenum = len(dictionary.token2id.keys())
print('featurenum', featurenum)

# 生成语料 
corpus = [dictionary.doc2bow(text) for text in textTest]
print('语料：', corpus)

""" 计算语料的TFIDF """ 
# 训练TFIDF模型 
tfidf_model = models.TfidfModel(corpus, dictionary=dictionary)
print('tfidf_model=', tfidf_model) # 只要记录BOW矩阵的非零元素个数(num_nnz)

# 得到语料的TFIDF值
corpus_tfidf = tfidf_model[corpus]
print('corpus=', corpus)
print('转换整个语料库：')
for doc in corpus_tfidf:
    print(doc)

# 生成余弦相似度索引, 使用SparseMatrixSimilarity()，可以占用更少的内存和磁盘空间。
index = similarities.SparseMatrixSimilarity(corpus_tfidf, num_features=featurenum) 



""" 对于新的句子，生成 BOW向量，与之前的句子计算相似度 """
test = jieba.lcut("我喜欢看电视剧")
print('分词', test)
# 生成BOW向量
vec = dictionary.doc2bow(test)
print('BOW向量', vec)
# 计算tfidf向量
test_vec = tfidf_model[vec]
print('test vec=', test_vec)
# 返回test_vec 和训练语料中所有文本的余弦相似度。返回结果是个numpy数组
print(index.get_similarities(test_vec))


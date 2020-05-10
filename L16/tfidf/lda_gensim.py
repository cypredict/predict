# Gensim, TF-IDF使用
import jieba
from gensim import corpora, models, similarities
import jieba.posseg as pseg

# 加载数据
wordslist = ["我非常喜欢看电视剧","我非常喜欢旅行","我非常喜欢吃苹果", '我非常喜欢跑步', '王者荣耀春季赛开战啦']

# 分词的时候不被切掉
jieba.add_word('王者荣耀', tag='n')
 
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

# 训练TFIDF模型 
tfidf_model = models.TfidfModel(corpus, dictionary=dictionary)
print('tfidf_model=', tfidf_model) # 只要记录BOW矩阵的非零元素个数(num_nnz)

# 得到每个单词的TF-IDF值
corpus_tfidf = tfidf_model[corpus]
print('corpus=', corpus)
print('转换整个语料库：')
for doc in corpus_tfidf:
    print(doc)

# 生成余弦相似度索引, 使用SparseMatrixSimilarity()，可以占用更少的内存和磁盘空间。
index = similarities.SparseMatrixSimilarity(corpus_tfidf, num_features=featurenum) 

# 测试阶段   模型对测试集进行operation；求余弦相似度。对于给定的新文本，找到训练集中最相似的五篇文章作为推荐
test = jieba.lcut("我喜欢看电视剧")
print(test)
# 生成BOW向量
vec = dictionary.doc2bow(test)
print(vec)

#生成tfidf向量
test_vec = tfidf_model[vec]
print('test_vec=', test_vec)
# 返回test_vec 和训练语料中所有文本的余弦相似度。返回结果是个numpy数组
print(index.get_similarities(test_vec))


""" 继续 LDA模型 """
lda = models.ldamodel.LdaModel(corpus = corpus, id2word=dictionary, num_topics=2)
for topic in lda.print_topics(num_words=5):
	print(topic)

# 主题推断
print(lda.inference(corpus))
text5 = '我喜欢看王者荣耀KPL挑战赛'
# bow 向量
bow = dictionary.doc2bow([word for word in jieba.cut(text5)])
print('bow:', bow)

inference_result = lda.inference([bow])[0]
print(text5)
for e, value in enumerate(inference_result[0]):
	print('主题{} 推断值{}\n'.format(e, value))

# 得到向量ID
word = '王者荣耀'
word_id = dictionary.doc2idx([word])[0]
print(word_id)

# 得到指定单词与主题的关系
print(lda.get_term_topics(word_id))
for i in lda.get_term_topics(word_id):
	print('{}与主题{}的关系值为{}%'.format(word, i[0], i[1]*100))

# 查看主题0的重要词汇
print(lda.get_topic_terms(0, topn=10))
# Gensim, TF-IDF使用
import jieba
from gensim import corpora, models, similarities
import jieba.posseg as jp, jieba

# 加载数据
wordslist = ["我非常喜欢看电视剧","我非常喜欢旅行","我非常喜欢吃苹果", '我非常喜欢跑步', '王者荣耀KPL春季赛开战啦'] 
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

print(dictionary.token2id.keys()) # 通过token2id得到特征
featurenum = len(dictionary.token2id.keys()) #token2id：词到id编码的映射, id2token：id编码到词的映射；
#dictionary.save('./my_dictionary.dict')


# 生成语料 
corpus = [dictionary.doc2bow(text) for text in textTest]
#print('语料：', corpus)
# 将生成的语料保存成MM文件
#corpora.MmCorpus.serialize('ths_corpuse.mm', corpus)  
# 加载
#corpus = corpora.MmCorpus('ths_corpuse.mm')  


# 训练TFIDF模型 
tfidf_model = models.TfidfModel(corpus, dictionary=dictionary)
print('tfidf_model=', tfidf_model) # 只要记录BOW矩阵的非零元素个数(num_nnz)
# 保存模型
#tfidf_model.save('my_model.tfidf')
# 载入模型
#tfidf_model = models.TfidfModel.load("my_model.tfidf")


# 用语料生成tfidf
corpus_tfidf = tfidf_model[corpus]
print('corpus=', corpus)
print('转换整个语料库：')
for doc in corpus_tfidf:
    print(doc)
#保存成model格式
#corpus_tfidf.save("ths_tfidf.model")
#加载
#corpus_tfidf = models.TfidfModel.load("ths_tfidf.model")
# 生成余弦相似度索引, 使用SparseMatrixSimilarity()，可以占用更少的内存和磁盘空间。
index = similarities.SparseMatrixSimilarity(corpus_tfidf, num_features=featurenum) 
index.save('train_index.index')
print(index)

# 测试阶段   模型对测试集进行operation；求余弦相似度。对于给定的新文本，找到训练集中最相似的五篇文章作为推荐

test = jieba.lcut("我喜欢看电视剧")
print(test)
# 生成BOW向量
vec = dictionary.doc2bow(test)
print(vec)
#生成tfidf向量
test_vec = tfidf_model[vec]
print('tfidf_model=', tfidf_model)
print('test_vec=', test_vec)
# 返回test_vec 和训练语料中所有文本的余弦相似度。返回结果是个numpy数组
print(index.get_similarities(test_vec))

"""训练LDA模型"""
print(corpus)
lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=2)
for topic in lda.print_topics(num_words=5):
    print(topic)

# 主题推断
print(lda.inference(corpus))

text5 = '我喜欢看王者荣耀KPL挑战赛'
# bow向量
bow = dictionary.doc2bow([word.word for word in jp.cut(text5)])
ndarray = lda.inference([bow])[0]
print(text5)
# enumerate() 函数用于将一个可遍历的数据对象
for e, value in enumerate(ndarray[0]):
    print('\t主题%d推断值%.2f' % (e, value))
    
# 得到向量ID
word = '王者荣耀'
word_id = dictionary.doc2idx([word])[0]
print(word_id)
# 得到指定单词 与主题的关系
print(lda.get_term_topics(word_id))
for i in lda.get_term_topics(word_id):
    print('【%s】与【主题%d】的关系值：%.2f%%' % (word, i[0], i[1]*100))

# 查看主题0 的重要词汇
print(lda.get_topic_terms(0, topn=10))
from gensim import corpora, models
import jieba.posseg as jp, jieba
# 文本集
texts = [
    '美国教练坦言，没输给中国女排，是输给了郎平',
    '美国无缘四强，听听主教练的评价',
    '中国女排晋级世锦赛四强，全面解析主教练郎平的执教艺术',
    '为什么越来越多的人买MPV，而放弃SUV？跑一趟长途就知道了',
    '跑了长途才知道，SUV和轿车之间的差距',
    '家用的轿车买什么好']
jieba.add_word('四强', 9, 'n')
flags = ('n', 'nr', 'ns', 'nt', 'eng', 'v', 'd')  # 词性
stopwords = ('没', '就', '知道', '是', '才', '听听', '坦言', '全面', '越来越', '评价', '放弃', '人') 
words_ls = []
for text in texts:
    words = [word.word for word in jp.cut(text) if word.flag in flags and word.word not in stopwords]
    words_ls.append(words)
# print(words_ls)
#去重，存到字典
dictionary = corpora.Dictionary(words_ls)
print('dictionary=', dictionary)
#print('dictionary=', dictionary.id2token)

corpus = [dictionary.doc2bow(words) for words in words_ls]
# print(corpus)
lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=2)
for topic in lda.print_topics(num_words=5):
    print(topic)

# 主题推断
print(lda.inference(corpus))

text5 = '中国女排将在郎平的率领下向世界女排三大赛的三连冠发起冲击'
# bow向量
bow = dictionary.doc2bow([word.word for word in jp.cut(text5) if word.flag in flags and word.word not in stopwords])
ndarray = lda.inference([bow])[0]
print(text5)
# enumerate() 函数用于将一个可遍历的数据对象
for e, value in enumerate(ndarray[0]):
    print('\t主题%d推断值%.2f' % (e, value))
    
# 得到向量ID
word = '轿车'
word_id = dictionary.doc2idx([word])[0]
print(word_id)
# 得到指定单词 与主题的关系
print(lda.get_term_topics(word_id))
for i in lda.get_term_topics(word_id):
    print('【%s】与【主题%d】的关系值：%.2f%%' % (word, i[0], i[1]*100))

# 查看主题0 的重要词汇
print(lda.get_topic_terms(0, topn=10))
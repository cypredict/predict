# 希拉里邮件分类
import numpy as np
import pandas as pd
import re
from gensim import corpora, models, similarities
import gensim
from nltk.corpus import stopwords
import jieba
 
 
def clean_email_text(text):
    # 换行，可以删除
    text = text.replace('\n'," ") 
    # 把 "-" 的两个单词，分开。（比如：july-edu ==> july edu）
    text = re.sub(r"-", " ", text) 
    # 日期，对主题模型没什么意义
    text = re.sub(r"\d+/\d+/\d+", "", text) 
    # 时间，没意义
    text = re.sub(r"[0-2]?[0-9]:[0-6][0-9]", "", text) 
    # 邮件地址，没意义
    text = re.sub(r"[\w]+@[\.\w]+", "", text) 
    # 网址，没意义
    text = re.sub(r"/[a-zA-Z]*[:\//\]*[A-Za-z0-9\-_]+\.+[A-Za-z0-9\.\/%&=\?\-_]+/i", "", text) 
    pure_text = ''
    # 以防还有其他特殊字符（数字）等等，我们直接把他们loop一遍，过滤掉
    for letter in text:
        # 只留下字母和空格
        if letter.isalpha() or letter==' ':
            pure_text += letter
    # 再把那些去除特殊字符后落单的单词，直接排除。
    # 我们就只剩下有意义的单词了。
    text = ' '.join(word for word in pure_text.split() if len(word)>1)
    return text
 
# 数据加载
df = pd.read_csv("./Emails.csv")
# 原邮件数据中有很多Nan的值，直接扔了。
df = df[['Id', 'ExtractedBodyText']].dropna()

docs = df['ExtractedBodyText']
#print(docs)
docs = docs.apply(lambda s: clean_email_text(s))

# 转化为列表List
doclist = docs.values

texts = [[word for word in jieba.cut(doc)] for doc in doclist]
#print(texts)

from gensim import corpora, models, similarities
"""第二步：构建语料库，将文本ID化"""
dictionary = corpora.Dictionary(texts)
print(dictionary)
# 把文档 doc变成一个稀疏向量
corpus = [dictionary.doc2bow(text) for text in texts]
# 将每一篇邮件ID化
print("第1封邮件ID化后的结果为：\n",corpus[0],'\n')

"""训练LDA模型"""
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10)# 所有主题的单词分布

# 每一行包含了 主题词和主题词权重
print(lda.print_topics(num_topics=10, num_words=10))

# topics中的重要单词
for topic in lda.print_topics(num_words=5):
    print(topic)

#print(corpus[0])
print(lda.get_document_topics(corpus[0]))
print(lda.get_document_topics(corpus[1]))
print(lda.get_document_topics(corpus[2]))

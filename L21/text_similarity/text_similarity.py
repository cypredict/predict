import re
import numpy as np
import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from tqdm import tqdm
from pprint import pprint
import os

# 加载停用词
with open('chinese_stopwords.txt','r', encoding='utf-8') as file:
    stopwords=[i[:-1] for i in file.readlines()]
#stopwords = [line.strip() for line in open('chinsesstoptxt.txt',encoding='UTF-8').readlines()]

# 数据加载
news = pd.read_csv('sqlResult.csv',encoding='gb18030')
print(news.shape)
print(news.head(5))


# 处理缺失值
print(news[news.content.isna()].head(5))
news=news.dropna(subset=['content'])
print(news.shape)


# 分词
def split_text(text):
    #return ' '.join([w for w in list(jieba.cut(re.sub('\s|[%s]' % (punctuation),'',text))) if w not in stopwords])
    text = text.replace(' ', '')
    text = text.replace('\n', '')
    text2 = jieba.cut(text.strip())
    result = ' '.join([w for w in text2 if w not in stopwords])
    return result

print(news.iloc[0].content)
print(split_text(news.iloc[0].content))

if not os.path.exists("corpus.pkl"):
    # 对所有文本进行分词
    corpus=list(map(split_text,[str(i) for i in news.content]))
    print(corpus[0])
    print(len(corpus))
    print(corpus[1])
    # 保存到文件，方便下次调用
    with open('corpus.pkl','wb') as file:
        pickle.dump(corpus, file)
else:
    # 调用上次处理的结果
    with open('corpus.pkl','rb') as file:
        corpus = pickle.load(file)



# 得到corpus的TF-IDF矩阵
countvectorizer = CountVectorizer(encoding='gb18030',min_df=0.015)
tfidftransformer = TfidfTransformer()

countvector = countvectorizer.fit_transform(corpus)
print(countvector.shape)

tfidf = tfidftransformer.fit_transform(countvector)
print(tfidf.shape)

# 标记是否为自己的新闻
label=list(map(lambda source: 1 if '新华' in str(source) else 0,news.source))
#print(label)

# 数据集切分
X_train, X_test, y_train, y_test = train_test_split(tfidf.toarray(), label, test_size = 0.3, random_state=42)
clf = MultinomialNB()
clf.fit(X=X_train, y=y_train)

"""
# 进行CV=3折交叉验证
scores=cross_validate(clf, X_train, y_train, scoring=('accuracy','precision','recall','f1'), cv=3, return_train_score=True)
pprint(scores)
"""
y_predict = clf.predict(X_test)
def show_test_reslt(y_true,y_pred):
    print('accuracy:',accuracy_score(y_true,y_pred))
    print('precison:',precision_score(y_true,y_pred))
    print('recall:',recall_score(y_true,y_pred))
    print('f1_score:',f1_score(y_true,y_pred))

show_test_reslt(y_test, y_predict)

# 使用模型检测抄袭新闻
prediction = clf.predict(tfidf.toarray())
labels = np.array(label)
# compare_news_index中有两列：prediction为预测，labels为真实值
compare_news_index = pd.DataFrame({'prediction':prediction,'labels':labels})
# copy_news_index：可能是Copy的新闻（即找到预测为1，但是实际不是“新华社”）
copy_news_index=compare_news_index[(compare_news_index['prediction'] == 1) & (compare_news_index['labels'] == 0)].index
# 实际为新华社的新闻
xinhuashe_news_index=compare_news_index[(compare_news_index['labels'] == 1)].index
print('可能为Copy的新闻条数：', len(copy_news_index))

if not os.path.exists("label.pkl"):
    # 使用k-means对文章进行聚类
    from sklearn.preprocessing import Normalizer
    from sklearn.cluster import KMeans
    normalizer = Normalizer()
    scaled_array = normalizer.fit_transform(tfidf.toarray())

    # 使用K-Means, 对全量文档进行聚类
    kmeans = KMeans(n_clusters=25,random_state=42,n_jobs=-1)
    k_labels = kmeans.fit_predict(scaled_array)
    # 保存到文件，方便下次调用
    with open('label.pkl','wb') as file:
        pickle.dump(k_labels, file)
    print(k_labels.shape)
    print(k_labels[0])
else:
    # 调用上次处理的结果
    with open('label.pkl','rb') as file:
        k_labels = pickle.load(file)

if not os.path.exists("id_class.pkl"):
    # 创建id_class
    id_class = {index:class_ for index, class_ in enumerate(k_labels)}
    # 保存到文件，方便下次调用
    with open('id_class.pkl','wb') as file:
        pickle.dump(id_class, file)
else:
    # 调用上次处理的结果
    with open('id_class.pkl','rb') as file:
        id_class = pickle.load(file)


if not os.path.exists("class_id.pkl"):
    from collections import defaultdict
    # 创建你class_id字段，key为classId,value为文档index
    class_id = defaultdict(set)
    for index,class_ in id_class.items():
        # 只统计新华社发布的class_id
        if index in xinhuashe_news_index.tolist():
            class_id[class_].add(index)
    # 保存到文件，方便下次调用
    with open('class_id.pkl','wb') as file:
        pickle.dump(class_id, file)
else:
    # 调用上次处理的结果
    with open('class_id.pkl','rb') as file:
        class_id = pickle.load(file)


# 输出每个类别的 文档个数
count=0
for k in class_id:
    print(count, len(class_id[k]))
    count +=1

# 查找相似文本（使用聚类结果进行filter）
def find_similar_text(cpindex, top=10):
    # 只在新华社发布的文章中查找
    dist_dict={i:cosine_similarity(tfidf[cpindex],tfidf[i]) for i in class_id[id_class[cpindex]]}
    # 从大到小进行排序
    return sorted(dist_dict.items(),key=lambda x:x[1][0], reverse=True)[:top]


import editdistance
# 指定某篇文章的相似度
#print(copy_news_index)
cpindex = 3352 # 在copy_news_index
#print('是否在新华社', cpindex in xinhuashe_news_index)
#print('是否在copy_news', cpindex in copy_news_index)
#print('3134是否在新华社', 3134 in xinhuashe_news_index)
#print('3134是否在copy_news', 3134 in copy_news_index)

#print(cpindex)
similar_list = find_similar_text(cpindex)
print(similar_list)
print('怀疑抄袭:\n', news.iloc[cpindex].content)
# 找一篇相似的原文
similar2 = similar_list[0][0]
print('相似原文:\n', news.iloc[similar2].content)
# 求任意两篇文章的编辑距离 
print('编辑距离:',editdistance.eval(corpus[cpindex], corpus[similar2]))


def find_similar_sentence(candidate, raw):
    similist = []
    cl = candidate.strip().split('。')
    ra = raw.strip().split('。')
    for c in cl:
        for r in ra:
            similist.append([c,r,editdistance.eval(c,r)])
    # 最相似的5个句子
    sort=sorted(similist,key=lambda x:x[2])[:5]
    for c,r,ed in sort:
        if c!='' and r!='':
            print('怀疑抄袭句:{0}\n相似原句:{1}\n 编辑距离:{2}\n'.format(c,r,ed))
# 查找copy文章 和第一相似的原文的比对
find_similar_sentence(news.iloc[cpindex].content, news.iloc[similar2].content)

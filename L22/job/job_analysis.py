import pandas as pd
# 数据加载
file = 'jobs_4k.xls'
content = pd.read_excel(file)
#print(content.head())
position_names = content['positionName'].tolist()
skill_lables = content['skillLables'].tolist()
#print(len(position_names))
#print(len(skill_lables))

# 图可视化
from collections import defaultdict
skill_position_graph = defaultdict(list)
for p, s in zip(position_names, skill_lables):
    #print(p, s)
    skill_position_graph[p] += eval(s)
    #skill_position_graph[p] += s
print(len(skill_position_graph))
# 技能字典，即 {职位：[技能1，技能2]}
#print(skill_position_graph)


import networkx as nx
import matplotlib.pyplot as plt
G = nx.Graph(skill_position_graph)
# 设置中文字体为黑体
plt.rcParams['font.sans-serif'] = ['SimHei']
# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False

# 以30个随机选取的工作岗位为例
import random 
# 从职位中随机选取30个
sample_nodes = random.sample(position_names, k=30)
sample_nodes_connections = sample_nodes
for p, skills in skill_position_graph.items():
    if p in sample_nodes: 
        sample_nodes_connections += skills
# 抽取G的节点作为子图
sample_graph = G.subgraph(sample_nodes_connections)
f = plt.figure(figsize=(50, 30))
pos=nx.spring_layout(sample_graph, k=1)
nx.draw(sample_graph, pos, with_labels=True, node_size=30, font_size=10)
plt.show()
nx.draw(sample_graph, pos, with_labels=True, node_size=40, font_size=30,  ax=f.add_subplot(111))
f.savefig('job-connections.png', dpi=180)
plt.show()


# 使用PageRank算法，对核心能力和核心职位排序
pr = nx.pagerank(G, alpha=0.9)
ranked_position_and_ability = sorted([(name, value) for name, value in pr.items()], 
                                     key=lambda x: x[1],
                                    reverse=True)
#print(ranked_position_and_ability)

# 根据计算结果，最容易找到工作的技能分别是 “后端”，“运维”，“Python”，“PHP”，“Java”，“UI”和“产品经理”
# 输入技能点，预测工资
#print(content.columns)
# 特征X去掉salary
X_content = content.drop(['salary'], axis=1)
# 目标Target
target = content['salary'].tolist()
#print(X_content.head())

string_training_corpus = []
# 将X_content内容都拼接成字符串，设置为merged字段
X_content['merged'] = X_content.apply(lambda x: ''.join(str(x)), axis=1)
#print(X_content['merged'][0]) # 字符串
#print(type(X_content['merged'][0]))
# 转换为list
X_string = X_content['merged'].tolist()
print(len(X_string))

import jieba
import re

# 合并到一起
def get_one_row_job_string(x_string_row):
    job_string = ''
    for i, element in enumerate(x_string_row.split('\n')):
        #print(element)  # id .............3
        # =2才正确, 因为是字典类型
        if len(element.split()) == 2:
            _, value = element.split()
            # i=0为id字段，不要
            if i == 0: continue
            #print(value)
            # 只保存value
            job_string += value
    return job_string

def token(string):
    return re.findall('\w+', string)

cutted_X = []
for i, row in enumerate(X_string):
    #print(row)
    #break
    job_string = get_one_row_job_string(row)
    #print(job_string)
    if i % 1000 == 0: print(i)
    cutted_X.append(' '.join(list(jieba.cut(''.join(token(job_string))))))
#print(cutted_X)


# 使用TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cutted_X)
print(target[:10]) #['10k-15k', '30k-50k', '40k-75k', '20k-40k', '20k-38k', '30k-50k', '40k-60k', '30k-50k', '40k-70k', '30k-50k']
#print(target)

import numpy as np
# 求平均值，薪资 10k-15k => 12.5k
target_numical = [np.mean(list(map(float, re.findall('\d+', s)))) for s in target]
Y = target_numical
#print(Y)

# 使用KNN模型
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=2)
model.fit(X, Y)
#print(neigh.score)

from sklearn.svm import SVR
model = SVR()
model.fit(X, Y)

# 贝叶线性回归
from sklearn.linear_model import BayesianRidge
model = BayesianRidge(compute_score=True)
model.fit(X.toarray(), Y)

def predicate_by_label(test_string, model):
    test_words = list(jieba.cut(test_string))
    test_vec = vectorizer.transform(test_words)
    predicated_value = model.predict(test_vec)
    return predicated_value[0]

test = '测试 北京 3年 专科'
print(test, predicate_by_label(test, model))
test2 = '测试 北京 4年 专科'
print(test2, predicate_by_label(test2, model))
test3 = '算法 北京 4年 本科'
print(test3, predicate_by_label(test3, model))
test4 = 'UI 北京 4年 本科'
print(test4, predicate_by_label(test4, model))

persons = [
    "广州Java本科3年掌握大数据",
    "沈阳Java硕士3年掌握大数据", 
    "沈阳Java本科3年掌握大数据", 
    "北京算法硕士3年掌握图像识别",
]
for p in persons:
    print('{} 的薪资AI预测结果是{}'.format(p, predicate_by_label(p, model)))

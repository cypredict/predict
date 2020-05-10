# 希拉里邮件内容清洗
import numpy as np
import pandas as pd
import re
from gensim import corpora, models, similarities
import gensim
from nltk.corpus import stopwords
 
 
# 对邮件正文进行清洗
def clean_email_text(text):
    # 换行，可以删除
    text = text.replace('\n'," ") 
    # 把 "-" 的两个单词，分开。（比如：meet-the-right ==> meet the right）
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
    # 检查每个字母，去掉其他特殊字符等
    for letter in text:
        # 只留下字母和空格
        if letter.isalpha() or letter==' ':
            pure_text += letter
    # 最终得到的都是有意义的单词
    text = ' '.join(word for word in pure_text.split() if len(word)>1)
    return text
 
# 数据加载
df = pd.read_csv("./Emails.csv")
# 原邮件数据中有很多Nan的值，直接扔了。
df = df[['Id', 'ExtractedBodyText']].dropna()

docs = df['ExtractedBodyText']
docs = docs.apply(lambda s: clean_email_text(s))
#print(docs)

# 转化为列表List
doclist = docs.values
print(doclist)
print(len(doclist))

import jieba

text = '张飞和关羽'
words = jieba.cut(text)
words = list(words)
print(words)

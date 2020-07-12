# 使用Python实现 reduce
import sys

countMap = {}
#temp = ['d\t1','e\t1'] 
#for line in temp:
for line in sys.stdin:
    line = line.strip()
    word, count = line.split('\t')
    try:
        count = int(count)
    except ValueError:  #count如果不是数字的话，直接忽略掉
        continue
    if word not in countMap:
        countMap[word] = count
    else:
        countMap[word] = countMap[word] + count

for key in countMap:
    print("%s\t%s" % (key, countMap[key]))

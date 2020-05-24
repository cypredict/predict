# 将数据集变成小样本
filename = 'cnews.train.txt'
num_cat = {}
num_max = 100

# 读取文件
contents, labels = [], []
with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        label, content = line.strip().split('\t')
        #print(label)
        if content:
            if label not in num_cat:
                num_cat[label] = 1
                contents.append(content)
                labels.append(label)
            else:
                if num_cat[label] < 100:
                    num_cat[label] = num_cat[label] + 1
                    contents.append(content)
                    labels.append(label)

# 写文件
with open('cnews.train.small.txt', 'w', encoding='utf-8', errors='ignore') as f:
    for content, label in zip(contents, labels):
        f.write(label + '\t' + content+'\n')
    f.close()
print(len(contents))
print(contents[0])
print(contents[1])
print(num_cat)


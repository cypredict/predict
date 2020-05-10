# 正则表达式实例
import re

# 在起始位置匹配
print(re.match('www', 'www.baidu.com').span())
print(re.match('com', 'www.baidu.com'))

#扫描整个字符串并返回第一个成功的匹配
print(re.search('www', 'www.baidu.com').span())
print(re.search('com', 'www.baidu.com').span())


phone = "001-609-7267 # ETS的TOEFL查询电话" 
# 删除字符串中的注释 
num = re.sub(r'#.*$', "", phone)
print("电话号码是: ", num)
 
# 删除非数字(-)的字符串 
num = re.sub(r'\D', "", phone)
print("电话号码是 : ", num)

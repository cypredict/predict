# 正则表达式实例
import re

def is_phone(phone):
	# 电话号码正则对象
	prog = re.compile('^1[35678]\d{9}$') 
	result = prog.match(phone)
	if result:
		return True
	else:
		return False
print(is_phone('15801234567'))

# 正则对象，匹配两个数字
prog = re.compile('\d{2}') 
# 下面两个作用是相同的
print(prog.search('12abc'))
print(re.search('\d{2}', '12abc'))
# 通过调用group()方法得到匹配的字符串,如果字符串没有匹配，则返回None
print(prog.search('12abc').group())


# group(0)永远是原始字符串，group(1)、group(2)……表示第1、2、……个子串
m = re.match(r'^(\d{3})-(\d{3,8})$', '010-12345')
print(m.group(0))
print(m.group(1))
print(m.group(2))
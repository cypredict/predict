# 使用正则表达式 判断电话号码是否正确
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
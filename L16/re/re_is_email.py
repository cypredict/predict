# 使用正则表达式 判断电话号码是否正确
import re

def is_email(email):
	# 邮箱正则对象
	# @之前可以 _  .  - 
	prog = re.compile('^[a-zA-Z0-9_.-]+@[a-zA-Z0-9-]+(\.[a-zA-Z0-9-]+)*\.[a-zA-Z0-9]{2,4}$') 
	result = prog.match(email)
	if result:
		return True
	else:
		return False

print(is_email("1580123@qq.com"))
print(is_email("1580123@qq"))
print(is_email("1580123@cc.ww.qq.yy.gg.wgh.com"))

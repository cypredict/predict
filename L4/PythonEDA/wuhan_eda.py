# 可视化EDA，最新数据来源：https://github.com/BlankerL/DXY-2019-nCoV-Data
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from matplotlib.font_manager import FontProperties


# 折线图
def line_chart(provinceName):
	# 数据准备
	result = clean_df[clean_df['provinceName']==provinceName]
	result = result.sort_values(by="province_confirmedCount" , ascending=True) 
	print(result)
	# 使用Matplotlib画折线图
	plt.plot(result['updateTime'], result['province_confirmedCount'])
	plt.show()
	# 使用Seaborn画折线图
	sns.lineplot(x="updateTime", y="province_confirmedCount", data=result)
	plt.show()

data = pd.read_csv('./DXYArea.csv')
df = data[['provinceName', 'province_confirmedCount', 'updateTime']]
df['updateTime'] = df['updateTime'].str[0:10]
clean_df = df.drop_duplicates(['provinceName', 'updateTime'], keep = 'first')
#print(clean_df)

# 折线图
line_chart('重庆市')
#line_chart('北京市')


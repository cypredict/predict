#!/usr/bin/env python
# coding: utf-8

# In[6]:


# 使用Prophet预测manning未来365天的页面流量
# 从2007年12月10日开始
import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# 读入数据集
df = pd.read_csv('./shanghai_index_1990_12_19_to_2020_03_12.csv')
# 修改列名 Timestamp => ds, Price => y
df.rename(columns={'Timestamp':'ds', 'Price':'y'}, inplace=True)
print(df.head())


# In[7]:


#print(df.tail())
# 拟合模型
model = Prophet()
model.fit(df)

# 构建待预测日期数据框，periods = 365 代表除历史数据的日期外再往后推 365 天
future = model.make_future_dataframe(periods=365)
#print(future.tail())

# 预测数据集
forecast = model.predict(future)
#print(forecast.columns)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# 展示预测结果
model.plot(forecast)
plt.show()


# In[8]:


# 预测的成分分析绘图，展示预测中的趋势、周效应和年度效应
model.plot_components(forecast)
print(forecast.columns)


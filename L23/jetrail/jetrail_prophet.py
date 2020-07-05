#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import numpy as np

# 数据加载
train = pd.read_csv('./train.csv')
print(train.head())


# In[37]:


# 转换为pandas中的日期格式
train['Datetime'] = pd.to_datetime(train.Datetime, format='%d-%m-%Y %H:%M')
# 将Datetime作为train的索引
train.index = train.Datetime
#去掉ID，hour, Datetime字段
train.drop(['ID', 'Datetime'], axis=1, inplace=True)

# 按天进行采样
daily_train = train.resample('D').sum()
print(daily_train)
daily_train['ds'] = daily_train.index
daily_train['y'] = daily_train.Count
daily_train.drop(['Count'], axis=1, inplace=True)
print(daily_train)


# In[26]:





# In[35]:


from fbprophet import Prophet
# 拟合porphet模型
m = Prophet(yearly_seasonality=True, seasonality_prior_scale=0.1)
m.fit(daily_train)
# 预测未来7个月，213天
future = m.make_future_dataframe(periods=213)
forecast = m.predict(future)
m.plot(forecast)

# 查看各个成分
m.plot_components(forecast)


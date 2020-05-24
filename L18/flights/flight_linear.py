import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#% matplotlib inline

# 加载数据
data = pd.read_csv("flights.csv")
print(data.head())

dataset_ori = data['passengers'].values.astype('float32')
plt.plot(dataset_ori)
plt.show()

# 线性回归
from sklearn import linear_model 
model = linear_model.LinearRegression()
y = data['passengers']
x = [[x] for x in range(1, len(y)+1)]
#print(x)
#print(y)
train_size = int(len(x) * 0.7)
train_x = x[:train_size]
train_y = y[:train_size]
model.fit(train_x, train_y)
y = model.predict(x)
#print(y)
plt.plot(x, y)
plt.plot(dataset_ori)
plt.show()

# 非线性回归
from sklearn.preprocessing import PolynomialFeatures
# 0-3次方
poly_reg = PolynomialFeatures(degree=3)
x_poly = poly_reg.fit_transform(x)
print(x_poly)
train_x_poly = x_poly[:train_size]
model = linear_model.LinearRegression()
model.fit(train_x_poly, train_y)
y = model.predict(x_poly)
#print(len(x))
#print(len(y))
plt.plot(x, y)
plt.plot(dataset_ori)
plt.show()

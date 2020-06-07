# MNIST手写数字分类（多种分类方法）
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# 加载数据
digits = load_digits()
data = digits.data
# 数据探索
print(data.shape)
# 查看第一幅图像
print(digits.images[0])
# 第一幅图像代表的数字含义
print(digits.target[0])
# 将第一幅图像显示出来
"""
plt.gray()
plt.title('Hand Written Digits')
plt.imshow(digits.images[0])
plt.show()
"""
# 分割数据，将25%的数据作为测试集，其余作为训练集
train_x, test_x, train_y, test_y = train_test_split(data, digits.target, test_size=0.25, random_state=33)

"""
# 采用Z-Score规范化
ss = preprocessing.StandardScaler()
train_ss_x = ss.fit_transform(train_x)
test_ss_x = ss.transform(test_x)
"""

# 创建LR分类器
model = LogisticRegression()
#model.fit(train_ss_x, train_y)
model.fit(train_x, train_y)
#predict_y=model.predict(test_ss_x)
predict_y=model.predict(test_x)
print('LR准确率: %0.4lf' % accuracy_score(predict_y, test_y))

# 创建GaussianNB分类器
model = GaussianNB()
model.fit(train_x, train_y)
predict_y=model.predict(test_x)
print('GaussianNB准确率: %0.4lf' % accuracy_score(predict_y, test_y))
print(model.class_prior_)
print(model.class_count_)
print(model.theta_)
print(model.sigma_)

# 创建MultinomialNB分类器
model = MultinomialNB()
model.fit(train_x, train_y)
predict_y=model.predict(test_x)
print('MultinomialNB准确率: %0.4lf' % accuracy_score(predict_y, test_y))
print(model.class_log_prior_)
print(model.intercept_)
print(model.feature_log_prob_)
print(model.coef_)
print(model.class_count_)
print(model.feature_count_)
#print(model.feature_count_)


# 创建BernoulliNB分类器
model = BernoulliNB()
model.fit(train_x, train_y)
predict_y=model.predict(test_x)
print('BernoulliNB准确率: %0.4lf' % accuracy_score(predict_y, test_y))

# 创建决策树分类器
model = DecisionTreeClassifier()
model.fit(train_x, train_y)
predict_y=model.predict(test_x)
print('决策树准确率: %0.4lf' % accuracy_score(predict_y, test_y))

# 创建随机森林分类器
model = RandomForestClassifier()
model.fit(train_x, train_y)
predict_y=model.predict(test_x)
print('随机森林准确率: %0.4lf' % accuracy_score(predict_y, test_y))

# 创建SVM分类器
model = SVC()
model.fit(train_x, train_y)
predict_y=model.predict(test_x)
print('SVC准确率: %0.4lf' % accuracy_score(predict_y, test_y))

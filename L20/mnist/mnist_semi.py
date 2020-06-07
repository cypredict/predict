import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import datasets
from sklearn.semi_supervised import label_propagation
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression

# 数据加载
digits = datasets.load_digits()
# 第一个数字
print(digits.data[0])
print(digits.target[0])
# 全部数据
X = digits.data
y = digits.target

n_total_samples = len(digits.data) # 1797
n_labeled_points = int(n_total_samples*0.1) #179

# 创建LR分类器
lr = LogisticRegression()
lr.fit(X[:n_labeled_points], y[:n_labeled_points])
# 对剩余90%数据进行预测
predict_y=lr.predict(X[n_labeled_points:])
true_y = y[n_labeled_points:] 
print("准确率", (predict_y == true_y).sum()/(len(true_y)))
print("-"*20)

# 使用半监督学习
# 复制一份y
y_train = np.copy(y)
# 把未标注的数据全部标记为-1，也就是后90%数据
y_train[n_labeled_points:] = -1 

# 使用标签传播模型，进行训练
lp_model = label_propagation.LabelSpreading(gamma=0.25, max_iter=5) 
lp_model.fit(X,y_train)
# 得到预测的标签
predict_y = lp_model.transduction_[n_labeled_points:] 
# 真实的标签
true_y = y[n_labeled_points:] 
print("预测标签", predict_y)
print("真实标签", true_y)
print("准确率", (predict_y == true_y).sum()/(len(true_y)))
cm = confusion_matrix(true_y, predict_y, labels = lp_model.classes_)
print("Confusion matrix", cm)

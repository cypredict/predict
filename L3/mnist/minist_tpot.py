# 使用TPOT自动机器学习工具对MNIST进行分类
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np

# 加载数据
digits = load_digits()
data = digits.data
print(digits.target)
X_train, X_test, y_train, y_test = train_test_split(digits.data.astype(np.float64),
    digits.target.astype(np.float64), train_size=0.75, test_size=0.25)

tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_mnist_pipeline.py')
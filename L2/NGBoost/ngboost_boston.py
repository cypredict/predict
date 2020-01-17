from ngboost import NGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 加载数据集
X, Y = load_boston(return_X_y = True)
# 切分训练集，测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
# 使用NGRegressor
ngb = NGBRegressor().fit(X_train, Y_train)
Y_preds = ngb.predict(X_test)
# 计算MSE
test_MSE = mean_squared_error(Y_preds, Y_test)
print('MSE', test_MSE)
# 计算NLL Negative Log Likelihood
Y_dists = ngb.pred_dist(X_test)
test_NLL = -Y_dists.logpdf(Y_test.flatten()).mean()
print('NLL', test_NLL)

# 使用ALS进行矩阵分解
from itertools import product, chain
from copy import deepcopy


class Matrix(object):
    def __init__(self, data):
        self.data = data
        self.shape = (len(data), len(data[0]))

    def row(self, row_no):
        """Get a row of the matrix.
        Arguments:
            row_no {int} -- Row number of the matrix.
        Returns:
            Matrix
        """

        return Matrix([self.data[row_no]])

    def col(self, col_no):
        """Get a column of the matrix.
        Arguments:
            col_no {int} -- Column number of the matrix.
        Returns:
            Matrix
        """
        m = self.shape[0]
        return Matrix([[self.data[i][col_no]] for i in range(m)])

    @property
    def is_square(self):
        """Check if the matrix is a square matrix.
        Returns:
            bool
        """

        return self.shape[0] == self.shape[1]

    @property
    def transpose(self):
        """Find the transpose of the original matrix.
        Returns:
            Matrix
        """

        data = list(map(list, zip(*self.data)))
        return Matrix(data)

    def _eye(self, n):
        """Get a unit matrix with shape (n, n).
        Arguments:
            n {int} -- Rank of unit matrix.
        Returns:
            list
        """

        return [[0 if i != j else 1 for j in range(n)] for i in range(n)]

    @property
    def eye(self):
        """Get a unit matrix with the same shape of self.
        Returns:
            Matrix
        """

        assert self.is_square, "The matrix has to be square!"
        data = self._eye(self.shape[0])
        return Matrix(data)

    def _gaussian_elimination(self, aug_matrix):
        """To simplify the left square matrix of the augmented matrix
        as a unit diagonal matrix.
        Arguments:
            aug_matrix {list} -- 2d list with int or float.
        Returns:
            list -- 2d list with int or float.
        """

        n = len(aug_matrix)
        m = len(aug_matrix[0])

        # From top to bottom.
        for col_idx in range(n):
            # Check if element on the diagonal is zero.
            if aug_matrix[col_idx][col_idx] == 0:
                row_idx = col_idx
                # Find a row whose element has same column index with
                # the element on the diagonal is not zero.
                while row_idx < n and aug_matrix[row_idx][col_idx] == 0:
                    row_idx += 1
                # Add this row to the row of the element on the diagonal.
                for i in range(col_idx, m):
                    aug_matrix[col_idx][i] += aug_matrix[row_idx][i]

            # Elimiate the non-zero element.
            for i in range(col_idx + 1, n):
                # Skip the zero element.
                if aug_matrix[i][col_idx] == 0:
                    continue
                # Elimiate the non-zero element.
                k = aug_matrix[i][col_idx] / aug_matrix[col_idx][col_idx]
                for j in range(col_idx, m):
                    aug_matrix[i][j] -= k * aug_matrix[col_idx][j]

        # From bottom to top.
        for col_idx in range(n - 1, -1, -1):
            # Elimiate the non-zero element.
            for i in range(col_idx):
                # Skip the zero element.
                if aug_matrix[i][col_idx] == 0:
                    continue
                # Elimiate the non-zero element.
                k = aug_matrix[i][col_idx] / aug_matrix[col_idx][col_idx]
                for j in chain(range(i, col_idx + 1), range(n, m)):
                    aug_matrix[i][j] -= k * aug_matrix[col_idx][j]

        # Iterate the element on the diagonal.
        for i in range(n):
            k = 1 / aug_matrix[i][i]
            aug_matrix[i][i] *= k
            for j in range(n, m):
                aug_matrix[i][j] *= k

        return aug_matrix

    def _inverse(self, data):
        """Find the inverse of a matrix.
        Arguments:
            data {list} -- 2d list with int or float.
        Returns:
            list -- 2d list with int or float.
        """

        n = len(data)
        unit_matrix = self._eye(n)
        aug_matrix = [a + b for a, b in zip(self.data, unit_matrix)]
        ret = self._gaussian_elimination(aug_matrix)
        return list(map(lambda x: x[n:], ret))

    @property
    def inverse(self):
        """Find the inverse matrix of self.
        Returns:
            Matrix
        """

        assert self.is_square, "The matrix has to be square!"
        data = self._inverse(self.data)
        return Matrix(data)

    def _row_mul(self, row_A, row_B):
        """Multiply the elements with the same subscript in both arrays and sum them.
        Arguments:
            row_A {list} -- 1d list with float or int.
            row_B {list} -- 1d list with float or int.
        Returns:
            float or int
        """

        return sum(x[0] * x[1] for x in zip(row_A, row_B))

    def _mat_mul(self, row_A, B):
        """An auxiliary function of the mat_mul function.
        Arguments:
            row_A {list} -- 1d list with float or int.
            B {Matrix}
        Returns:
            list -- 1d list with float or int.
        """

        row_pairs = product([row_A], B.transpose.data)
        return [self._row_mul(*row_pair) for row_pair in row_pairs]

    def mat_mul(self, B):
        """Matrix multiplication.
        Arguments:
            B {Matrix}
        Returns:
            Matrix
        """

        error_msg = "A's column count does not match B's row count!"
        assert self.shape[1] == B.shape[0], error_msg
        return Matrix([self._mat_mul(row_A, B) for row_A in self.data])

    def _mean(self, data):
        """Calculate the average of all the samples.
        Arguments:
            X {list} -- 2d list with int or float.
        Returns:
            list -- 1d list with int or float.
        """

        m = len(data)
        n = len(data[0])
        ret = [0 for _ in range(n)]
        for row in data:
            for j in range(n):
                ret[j] += row[j] / m
        return ret

    def mean(self):
        """Calculate the average of all the samples.
        Returns:
            Matrix
        """

        return Matrix(self._mean(self.data))

    def scala_mul(self, scala):
        """Scala multiplication.
        Arguments:
            scala {float}
        Returns:
            Matrix
        """

        m, n = self.shape
        data = deepcopy(self.data)
        for i in range(m):
            for j in range(n):
                data[i][j] *= scala
        return Matrix(data)


import pandas as pd
import numpy as np
import random
from collections import defaultdict

class ALS(object):
    def __init__(self):
        self.user_ids = None
        self.item_ids = None
        self.user_ids_dict = None
        self.item_ids_dict = None
        self.user_matrix = None
        self.item_matrix = None
        self.user_items = None
        self.shape = None
        self.rmse = None


    def _process_data(self, X):
        """ 将评分矩阵X转化为稀疏矩阵
            输入参数X:
                X {list} -- 2d list with int or float(user_id, item_id, rating)
            输出结果:
                dict -- {user_id: {item_id: rating}}
                dict -- {item_id: {user_id: rating}}
        """        
        self.user_ids = tuple((set(map(lambda x: x[0], X))))
        self.user_ids_dict = dict(map(lambda x: x[::-1], enumerate(self.user_ids)))
     
        self.item_ids = tuple((set(map(lambda x: x[1], X))))
        self.item_ids_dict = dict(map(lambda x: x[::-1], enumerate(self.item_ids)))
     
        self.shape = (len(self.user_ids), len(self.item_ids))
     
        ratings = defaultdict(lambda: defaultdict(int))
        ratings_T = defaultdict(lambda: defaultdict(int))
        for row in X:
            user_id, item_id, rating = row
            ratings[user_id][item_id] = rating
            ratings_T[item_id][user_id] = rating
     
        err_msg = "Length of user_ids %d and ratings %d not match!" % (len(self.user_ids), len(ratings))
        assert len(self.user_ids) == len(ratings), err_msg
     
        err_msg = "Length of item_ids %d and ratings_T %d not match!" % (len(self.item_ids), len(ratings_T))
        assert len(self.item_ids) == len(ratings_T), err_msg
        return ratings, ratings_T

     
    def _users_mul_ratings(self, users, ratings_T):
        """ 用户矩阵(稠密) 与 评分矩阵（稀疏）相乘
            The result(items) is a k * n matrix, n stands for number of item_ids.
            Arguments:
                users {Matrix} -- k * m matrix, m stands for number of user_ids.
                ratings_T {dict} -- The items ratings by users.
                {item_id: {user_id: rating}}
            Returns:
                Matrix -- Item matrix.
        """
        def f(users_row, item_id):
            user_ids = iter(ratings_T[item_id].keys())
            scores = iter(ratings_T[item_id].values())
            col_nos = map(lambda x: self.user_ids_dict[x], user_ids)
            _users_row = map(lambda x: users_row[x], col_nos)
            return sum(a * b for a, b in zip(_users_row, scores))
     
        ret = [[f(users_row, item_id) for item_id in self.item_ids] for users_row in users.data]
        return Matrix(ret)

    def _items_mul_ratings(self, items, ratings):
        """ item矩阵（稠密）与评分矩阵（稀疏）相乘
        The result(users) is a k * m matrix, m stands for number of user_ids.
        Arguments:
            items {Matrix} -- k * n matrix, n stands for number of item_ids.
            ratings {dict} -- The items ratings by users.
            {user_id: {item_id: rating}}
        Returns:
            Matrix -- User matrix.
        """
        def f(items_row, user_id):
            item_ids = iter(ratings[user_id].keys())
            scores = iter(ratings[user_id].values())
            col_nos = map(lambda x: self.item_ids_dict[x], item_ids)
            _items_row = map(lambda x: items_row[x], col_nos)
            return sum(a * b for a, b in zip(_items_row, scores))
     
        ret = [[f(items_row, user_id) for user_id in self.user_ids] for items_row in items.data]
        return Matrix(ret)

    # 生成随机矩阵
    def _gen_random_matrix(self, n_rows, n_colums):
        #print(n_colums, ' ', n_rows)
        #data = [[random() for _ in range(n_colums)] for _ in range(n_rows)]
        #d = 2
        data = np.random.rand(n_rows, n_colums)
        return Matrix(data)


    # 计算RMSE
    def _get_rmse(self, ratings):
            m, n = self.shape
            mse = 0.0
            n_elements = sum(map(len, ratings.values()))
            for i in range(m):
                for j in range(n):
                    user_id = self.user_ids[i]
                    item_id = self.item_ids[j]
                    rating = ratings[user_id][item_id]
                    if rating > 0:
                        user_row = self.user_matrix.col(i).transpose
                        item_col = self.item_matrix.col(j)
                        rating_hat = user_row.mat_mul(item_col).data[0][0]
                        square_error = (rating - rating_hat) ** 2
                        mse += square_error / n_elements
            return mse ** 0.5

    # 模型训练
    def fit(self, X, k, max_iter=10):
        ratings, ratings_T = self._process_data(X)
        self.user_items = {k: set(v.keys()) for k, v in ratings.items()}
        m, n = self.shape
     
        error_msg = "Parameter k must be less than the rank of original matrix"
        assert k < min(m, n), error_msg
     
        self.user_matrix = self._gen_random_matrix(k, m)
     
        for i in range(max_iter):
            if i % 2:
                items = self.item_matrix
                self.user_matrix = self._items_mul_ratings(
                    items.mat_mul(items.transpose).inverse.mat_mul(items),
                    ratings
                )
            else:
                users = self.user_matrix
                self.item_matrix = self._users_mul_ratings(
                    users.mat_mul(users.transpose).inverse.mat_mul(users),
                    ratings_T
                )
            rmse = self._get_rmse(ratings)
            print("Iterations: %d, RMSE: %.6f" % (i + 1, rmse))
     
        self.rmse = rmse

    # Top-n推荐，用户列表：user_id, n_items: Top-n
    def _predict(self, user_id, n_items):
        users_col = self.user_matrix.col(self.user_ids_dict[user_id])
        users_col = users_col.transpose
     
        items_col = enumerate(users_col.mat_mul(self.item_matrix).data[0])
        items_scores = map(lambda x: (self.item_ids[x[0]], x[1]), items_col)
        viewed_items = self.user_items[user_id]
        items_scores = filter(lambda x: x[0] not in viewed_items, items_scores)
     
        return sorted(items_scores, key=lambda x: x[1], reverse=True)[:n_items]

    # 预测多个用户
    def predict(self, user_ids, n_items=10):
        return [self._predict(user_id, n_items) for user_id in user_ids]

def format_prediction(item_id, score):
    return "item_id:%d score:%.2f" % (item_id, score)

def load_movie_ratings(file_name):
    f = open(file_name)
    lines = iter(f)
    col_names = ", ".join(next(lines)[:-1].split(",")[:-1])
    print("The column names are: %s." % col_names)
    data = [[float(x) if i == 2 else int(x)
             for i, x in enumerate(line[:-1].split(",")[:-1])]
            for line in lines]
    f.close()

    return data

print("使用ALS算法") 
model = ALS()
# 数据加载
X = load_movie_ratings('./ratings_small.csv')
# 运行max_iter次
model.fit(X, k=3, max_iter=2)
"""
X = np.array([[1,1,1], [1,2,1], [2,1,1], [2,3,1], [3,2,1], [3,3,1], [4,1,1], [4,2,1],
              [5,4,1], [5,5,1], [6,4,1], [6,6,1], [7,5,1], [7,6,1], [8,4,1], [8,5,1], [9,4,1], [9,5,1],
              [10,7,1], [10,8,1], [11,8,1], [11,9,1], [12,7,1], [12,9,1]])
# 运行max_iter次
model.fit(X, k=3, max_iter=20)
"""

print("对用户进行推荐")
user_ids = range(1, 13)
# 对用户列表user_ids，进行Top-n推荐
predictions = model.predict(user_ids, n_items=2)
print(predictions)
for user_id, prediction in zip(user_ids, predictions):
    _prediction = [format_prediction(item_id, score) for item_id, score in prediction]
    print("User id:%d recommedation: %s" % (user_id, _prediction))



import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt

# 城市的数量
num_points = 50

# 生成点坐标
points_coordinate = np.random.rand(num_points, 2)
# 计算两点之间的距离
distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')
print(distance_matrix)


# 目标函数，输入路径 返回总距离
# 使用方式：compute_distance(np.arange(num_points))
def compute_distance(routine):
    num_points, = routine.shape
    #print(routine)
    #print(routine.shape)
    # 求和
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])


# 遗传算法
from sko.GA import GA_TSP
# prob_mut 种群的新成员由变异而非交叉得来的概率
ga_tsp = GA_TSP(func=compute_distance, n_dim=num_points, size_pop=50, max_iter=500, prob_mut=0.2)
best_points, best_distance = ga_tsp.run()

# 画图
fig, ax = plt.subplots(1, 2)
best_points_ = np.concatenate([best_points, [best_points[0]]])
best_points_coordinate = points_coordinate[best_points_, :]
ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
ax[1].plot(ga_tsp.generation_best_Y)
plt.show()

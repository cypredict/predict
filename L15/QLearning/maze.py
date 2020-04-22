# 模拟迷宫问题
import numpy as np
GAMMA = 0.8
Q = np.zeros((6,6))
R=np.asarray([[-1,-1,-1,-1,0,-1],
   [-1,-1,-1,0,-1,100],
   [-1,-1,-1,0,-1,-1],
   [-1,0, 0, -1,0,-1],
   [0,-1,-1,0,-1,100],
   [-1,0,-1,-1,0,100]])

# 取每一行的最大值
def getMaxQ(state):
  # 通过选取最大动作值来进行最优策略学习
  return max(Q[state, :])

# QLearning
def QLearning(state):
  curAction = None
  for action in range(6):
    if(R[state][action] == -1):
      Q[state, action] = 0
    else:
      curAction = action
      Q[state, action] = R[state][action] + GAMMA * getMaxQ(curAction)

for count in range(1000):
  for i in range(6):
    QLearning(i)
print(Q/5)
# 实现了蒙特卡洛树搜索 MCTS

import numpy as np
import copy
from operator import itemgetter


# 快速走子策略：随机走子
def rollout_policy_fn(board):
    # 随机走，从棋盘中可以下棋的位置中随机选一个
    action_probs = np.random.rand(len(board.availables))
    return zip(board.availables, action_probs)


# policy_value_fn 考虑了棋盘状态，输出一组(action, probability)和分数[-1,1]之间
def policy_value_fn(board):
    # 对于pure MCTS来说，返回统一的概率，得分score为0
    action_probs = np.ones(len(board.availables))/len(board.availables)
    return zip(board.availables, action_probs), 0

# MCTS树节点，每个节点都记录了自己的Q值，先验概率P和 访问计数调整前的得分（visit-count-adjusted prior score） u
class TreeNode(object):
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p
    # Expand，展开叶子节点（新的孩子节点），action_priors为(action, prior probability)
    def expand(self, action_priors):
        for action, prob in action_priors:
            # 如果不是该节点的子节点，那么就expand 添加为子节点
            if action not in self._children:
                # 父亲节点为当前节点self,先验概率为prob
                self._children[action] = TreeNode(self, prob)
    
    # Select步骤，在孩子节点中，选择具有最大行动价值UCT，通过get_value(c_puct)函数得到
    def select(self, c_puct):
        # 每次选择最大UCT值的节点，返回(action, next_node)
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    # 从叶子评估中，更新节点值，leaf_value表明了当前player的子树评估值
    def update(self, leaf_value):
        # 节点访问次数+1
        self._n_visits += 1
        # 更新Q值，Update Q, a running average of values for all visits.
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits
        
    # 递归的更新所有祖先，调用self.update
    def update_recursive(self, leaf_value):
        # 如果不是根节点，就需要先调用父亲节点的更新
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    # 计算节点价值 UCT值 = Q值 + 调整后的访问次数（exploitation + exploration）
    def get_value(self, c_puct):
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    # 判断是否为叶子节点
    def is_leaf(self):
        return self._children == {}
    
    # 判断是否为根节点
    def is_root(self):
        return self._parent is None

# MCTS：Monte Carlo Tree Search 实现了蒙特卡洛树的搜索 
class MCTS(object):
    # policy_value_fn 考虑了棋盘状态，输出一组(action, probability)和分数[-1,1]之间(预计结束时的比分期望)
    # c_puct exploitation和exploration之间的折中系数
    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        self._root = TreeNode(None, 1.0) # 根节点
        self._policy = policy_value_fn   # 策略状态，考虑了棋盘状态，输出一组(action, probability)和分数[-1,1]之间
        self._c_puct = c_puct # exploitation和exploration之间的折中系数
        self._n_playout = n_playout

    # 从根节点到叶节点运行每一个playout，获取叶节点的值（胜负平结果1，-1,0），并通过其父节点将其传播回来
    # 状态是就地修改的，所以需要保存副本
    def _playout(self, state):
        node = self._root
        while(1):
            if node.is_leaf():
                break
            # 基于贪心算法 选择下一步
            action, node = node.select(self._c_puct)
            state.do_move(action)
        # 对于current player，根据state 得到一组(action, probability)，这里不需要得到得分 _
        action_probs, _ = self._policy(state)
        # 检查游戏是否结束
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        # 采用快速走子策略，评估叶子结点值（是否获胜）
        leaf_value = self._evaluate_rollout(state)
        # 更新本次传播路径（遍历节点）中的（节点值 和 访问次数）
        node.update_recursive(-leaf_value)

    # 使用rollout策略，一直到游戏结束，如果当前选手获胜返回+1，对手获胜返回-1，平局返回0
    def _evaluate_rollout(self, state, limit=1000):
        player = state.get_current_player()
        for i in range(limit):
            end, winner = state.game_end()
            if end:
                break
            # 采用快速走子策略，得到action
            action_probs = rollout_policy_fn(state)
            max_action = max(action_probs, key=itemgetter(1))[0]
            state.do_move(max_action)
        else:
            # 如果没有break for循环，发出警告
            print("WARNING: rollout reached move limit")
        if winner == -1:  # 平局
            return 0
        else:
            return 1 if winner == player else -1
        
    # 顺序执行所有的playouts，输入的state为当前游戏的状态，返回最经常访问的action
    def get_move(self, state):
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        return max(self._root._children.items(),
                   key=lambda act_node: act_node[1]._n_visits)[0]

    # 在树中前进一步
    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"

# 基于MCTS的AI Player
class MCTSPlayer(object):
    def __init__(self, c_puct=5, n_playout=2000):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)
        
    # 设置player index
    def set_player_ind(self, p):
        self.player = p
        
    # 重置MCTS树
    def reset_player(self):
        self.mcts.update_with_move(-1)
    
    # 获取AI下棋的位置
    def get_action(self, board):
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            move = self.mcts.get_move(board)
            self.mcts.update_with_move(-1)
            return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)

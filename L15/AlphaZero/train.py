# 训练五子棋AI

from __future__ import print_function
import random
import numpy as np
# deque 是一个双端队列
from collections import defaultdict, deque
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure # 随机走子策略的AI
from mcts_alphaZero import MCTSPlayer # AlphaGo方式的AI
from policy_value_net_pytorch import PolicyValueNet  # Pytorch

class TrainPipeline():
    def __init__(self, init_model=None):
        # 设置棋盘和游戏的参数
        self.board_width = 6
        self.board_height = 6
        self.n_in_row = 4
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)
        # 设置训练参数
        self.learn_rate = 2e-3 # 基准学习率
        self.lr_multiplier = 1.0  # 基于KL自动调整学习倍速
        self.temp = 1.0  # 温度参数
        self.n_playout = 400  # 每下一步棋，模拟的步骤数
        self.c_puct = 5 # exploitation和exploration之间的折中系数
        self.buffer_size = 10000
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size) #使用 deque 创建一个双端队列
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02 # 早停检查
        self.check_freq = 50 # 每50次检查一次，策略价值网络是否更新
        self.game_batch_num = 500 # 训练多少个epoch
        self.best_win_ratio = 0.0 # 当前最佳胜率，用他来判断是否有更好的模型
        # 弱AI（纯MCTS）模拟步数，用于给训练的策略AI提供对手
        self.pure_mcts_playout_num = 1000
        if init_model:
            # 通过init_model设置策略网络
            self.policy_value_net = PolicyValueNet(self.board_width, self.board_height, model_file=init_model)
        else:
            # 训练一个新的策略网络
            self.policy_value_net = PolicyValueNet(self.board_width, self.board_height)
        # AI Player，设置is_selfplay=1 自我对弈，因为是在进行训练
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)
        
    # 通过旋转和翻转增加数据集, play_data: [(state, mcts_prob, winner_z), ..., ...]
    def get_equi_data(self, play_data):
        extend_data = []
        for state, mcts_porb, winner in play_data:
            # 在4个方向上进行expand，每个方向都进行旋转，水平翻转
            for i in [1, 2, 3, 4]:
                # 逆时针旋转
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
                # 水平翻转
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
        return extend_data

    # 收集自我对弈数据，用于训练
    def collect_selfplay_data(self, n_games=1):
        for i in range(n_games):
            # 与MCTS Player进行对弈
            winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
            play_data = list(play_data)[:]
            # 保存下了多少步
            self.episode_len = len(play_data)
            # 增加数据 play_data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)
            
    # 更新策略网络
    def policy_update(self):
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        # 保存更新前的old_probs, old_v
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            # 每次训练，调整参数，返回loss和entropy
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate*self.lr_multiplier)
            # 输入状态，得到行动的可能性和状态值，按照batch进行输入
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            # 计算更新前后两次的loss差
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # 动态调整学习倍率 lr_multiplier
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    # 用于评估训练网络的质量，评估一共10场play，返回比赛胜率（赢1分、输0分、平0.5分）
    def policy_evaluate(self, n_games=10):
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct, n_playout=self.n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5, n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            # AI和弱AI（纯MCTS）对弈，不需要可视化 is_shown=0，双方轮流职黑 start_player=i % 2
            winner = self.game.start_play(current_mcts_player, pure_mcts_player, start_player=i % 2, is_shown=0)
            win_cnt[winner] += 1
        # 计算胜率，平手计为0.5分
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    def run(self):
        # 开始训练
        try:
            # 训练game_batch_num次，每个batch比赛play_batch_size场
            for i in range(self.game_batch_num):
                # 收集自我对弈数据
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(i+1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                # 判断当前模型的表现，保存最优模型
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    win_ratio = self.policy_evaluate()
                    # 保存当前策略
                    self.policy_value_net.save_model('./current_policy.model')
                    if win_ratio > self.best_win_ratio:
                        print("发现新的最优策略，进行策略更新")
                        self.best_win_ratio = win_ratio
                        # 更新最优策略
                        self.policy_value_net.save_model('./best_policy.model')
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run()

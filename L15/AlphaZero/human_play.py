# 调用AI与人下五子棋
from __future__ import print_function
import pickle
from game import Board, Game #定义了棋盘Board
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet  # Pytorch


# 由人来输入下棋的位置
class Human(object):
    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    # 通过input交互，得到用户的下棋位置 move
    def get_action(self, board):
        try:
            location = input("输入你下棋的位置 x,y: ")
            print(location)
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("输入位置非法")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)


# GoBang主程序
def run():
    n = 5
    # 这里可以修改棋盘的大小，需要和AI Model的棋盘大小相等
    width, height = 6, 6
    # 调用AI模型
    model_file = 'best_policy.model'
    try:
        # 初始化棋盘
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        # ############### human VS AI ###################
        # 加载AI Model
        best_policy = PolicyValueNet(width, height, model_file = model_file, use_gpu=True)
        # 设置n_playout越大，效果越好，不需要设置is_selfplay，因为不需要进行AI训练
        mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)  

        # 也可以使用MCTS_Pure进行对弈，但是它太弱了
        # mcts_player = MCTS_Pure(c_puct=5, n_playout=1000)

        # 创建人类player, 输入下棋位置比如 3,3
        human = Human()

        # start_player=1表示电脑先手，0表示人先手
        game.start_play(human, mcts_player, start_player=1, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()

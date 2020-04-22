# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np


# 定义棋盘
class Board(object):
    def __init__(self, **kwargs):
        # 默认宽度、高度为8
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        # 保存棋盘状态，为字典结构 key为棋盘位置move，value为player编号
        self.states = {}
        # 设置 n子棋，默认为5
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2

    # 初始化棋盘，n_in_row子棋
    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        # 初始化current_player，设置为start player
        self.current_player = self.players[start_player]  
        # 保存棋盘中可以下棋的位置 list类型
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    # 通过move，返回location:h,w
    def move_to_location(self, move):
        h = move // self.width
        w = move % self.width
        return [h, w]

    # 输入location二维数组h,w，返回move
    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    # 返回当前用户的棋盘状态，状态大小为4*width*height
    def current_state(self):
        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            # 获取每一步，以及下棋的player
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            # 当前player状态
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            # 对手player状态
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # 记录最后一步（落子）的位置
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # 显示的颜色值 
        return square_state[:, ::-1, :]

    # 当前current_player下了一步棋，需要保存状态，执棋方切换
    def do_move(self, move):
        # 保存当前move 是由current_player下的
        self.states[move] = self.current_player
        # 下了一步棋，棋盘中可以下的位置就少了一个
        self.availables.remove(move)
        # 执棋方切换
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move

    # 判断是否有人获胜
    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        # 单方下棋步骤不足n_in_row
        if len(moved) < self.n_in_row *2-1:
            return False, -1

        for m in moved:
            # 将move转化为 [h,w]
            h = m // width
            w = m % width
            # 当前步是由哪个player下的
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    # 判断游戏是否结束
    def game_end(self):
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player

# Game Server
class Game(object):
    def __init__(self, board, **kwargs):
        self.board = board

    # 绘制棋盘和棋子信息
    def graphic(self, board, player1, player2):
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    # 开始比赛，player1与player2
    def start_play(self, player1, player2, start_player=0, is_shown=1):
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        # 初始化棋盘
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        # 设置player index
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        # 一直循环到比赛结束    
        while True:
            # 获取当前的player
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("游戏结束，获胜方为 ", players[winner])
                    else:
                        print("游戏结束，双方平局")
                return winner

    # AI自我对弈，存储自我对弈数据 用于训练 self-play data: (state, mcts_probs, z)
    def start_self_play(self, player, is_shown=0, temp=1e-3):
        # 初始化棋盘
        self.board.init_board()
        p1, p2 = self.board.players
        # 记录该局对应的数据：states, mcts_probs, current_players
        states, mcts_probs, current_players = [], [], []
        # 一直循环到比赛结束
        while True:
            # 得到player的下棋位置
            move, move_probs = player.get_action(self.board, temp=temp, return_prob=1)
            # 存储数据
            states.append(self.board.current_state()) #棋盘状态
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # 按照move来下棋
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            # 判断游戏是否结束end，统计获胜方 winner
            end, winner = self.board.game_end()
            if end:
                # 记录该局对弈中的每步分值，胜1，负-1，平局0
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # 重置MCTS根节点 reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("游戏结束，获胜一方为 ", winner)
                    else:
                        print("游戏结束，双方平局")
                # 返回获胜方，self-play数据: (state, mcts_probs, z)
                return winner, zip(states, mcts_probs, winners_z)

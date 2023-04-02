import numpy as np
import copy
from collections import namedtuple
from renju_rule import Renju_Rule
from player import Player
from pprint import pprint

class Point(namedtuple('Point', 'row col')):
    def __init__(self, row, col):
        pass
    # def neighbors(self):
    #     return [
    #         Point(self.row - 1, self.col),
    #         Point(self.row + 1, self.col),
    #         Point(self.row, self.col - 1),
    #         Point(self.row, self.col + 1),
    #     ]

    # def __deepcopy__(self, memodict={}):
    #     return self

class NoPossibleMove():
    def __init__(self):
        pass

class Board():
    def __init__(self, board_size):
        self.board_size = board_size
        self.grid = []
        self.init_grid()

    def init_grid(self):
        for r in range(self.board_size):
            row = []
            for c in range(self.board_size):
                row.append(0)
            self.grid.append(row)

    def is_empty(self, point):
        return self.grid[point.row][point.col]==0

    def get(self, point):
        return self.grid[point.row][point.col]
    
    def place_stone(self, player, point):
        self.grid[point.row][point.col] = player

    def is_on_grid(self, point):
        return 0 <= point.row < self.board_size and 0 <= point.col < self.board_size

    def is_full(self):
        for row in self.grid:
            for point in row:
                if point == 0:
                    return False
        return True
    

class GameState():
    def __init__(self, board, next_player, previous, move):
        self.board = board
        self.next_player = next_player
        self.previous_state = previous
        self.last_move = move
        self.rule = Renju_Rule(self.board)
        self.win_by_forcing_forbidden_move = False

    def apply_move(self, move):
        if isinstance(move, NoPossibleMove):
            self.win_by_forcing_forbidden_move = True
            self.winner = self.prev_player()
            return self
        next_board = copy.deepcopy(self.board)
        next_board.place_stone(self.next_player, move)
        return GameState(next_board, self.next_player.other, self, move)

    # one-man play. for checking applied rules.
    def apply_move_test(self, move):
        next_board = copy.deepcopy(self.board)
        next_board.place_stone(self.next_player, move)
        return GameState(next_board, self.next_player, self, move)

    def new_game(board_size: int):
        board = Board(board_size)
        return GameState(board, Player.black, None, None)

    def is_valid_move(self, move):
        if self.next_player == Player.black:
            if self.rule.forbidden_point(move.col, move.row, Player.black):
                return False
        return True    

    def legal_moves(self):
        moves = []
        for row in range(0, self.board.board_size):
            for col in range(0, self.board.board_size):
                move = Point(row, col)
                if not self.board.is_empty(move):
                    continue
                if self.is_valid_move(move):
                    moves.append(move)
        return moves
        
    def is_over(self):
        if self.previous_state is None:
            return False
        if self.board.is_full():
            self.winner = None
            return True
        if self.win_by_forcing_forbidden_move:
            return True
        if self.check_five():
            self.winner = self.prev_player()
            return True

    def check_five(self):
        return self.rule.is_five(self.last_move.col, self.last_move.row, self.prev_player())
    
    def prev_player(self):
        if self.previous_state is None:
            return None
        return self.previous_state.next_player

    # def check_win(self):
    #     if self.previous_state is None:
    #         return False        
        
    #     # 가로
    #     for r in range(self.board.board_size):
    #         cnt = 0
    #         for c in range(self.board.board_size):
    #             if self.board.get(Point(row=r, col=c))==self.prev_player():
    #                 cnt += 1
    #                 if cnt==5:
    #                     return True
    #             else:
    #                 cnt = 0
                    
    #             if (self.board.board_size - c - 1) + cnt < 5: # 남은 칸 더 안 봐도 됨
    #                 # print(r,c,cnt,(self.board.board_size - c - 1) + cnt,'break')
    #                 break

    #     # 세로
    #     for c in range(self.board.board_size):
    #         cnt = 0
    #         for r in range(self.board.board_size):
    #             if self.board.get(Point(row=r, col=c))==self.prev_player():
    #                 cnt += 1
    #                 if cnt==5:
    #                     return True
    #             else:
    #                 cnt = 0
    #             if (self.board.board_size - r - 1)  + cnt < 5: # 남은 칸 더 안 봐도 됨
    #                 break

    #     # 대각선
    #     # \ 방향
    #     startings = []
    #     for c in range(0, self.board.board_size - 4):
    #         startings.append((0,c))
    #     for r in range(1, self.board.board_size - 4):
    #         startings.append((r,0))
    #     for starting in startings:
    #         r, c = starting
    #         cnt = 0
    #         while r < self.board.board_size and c < self.board.board_size:
    #             if self.board.get(Point(row=r, col=c))==self.prev_player():
    #                 cnt += 1
    #                 if cnt == 5:
    #                     return True
    #             else:
    #                 cnt = 0
    #             r += 1
    #             c += 1

    #     # / 방향
    #     startings = []
    #     for c in range(4, self.board.board_size):
    #         startings.append((0,c))
    #     for r in range(1, self.board.board_size - 4):
    #         startings.append((r, self.board.board_size - 1))
    #     for starting in startings:
    #         r, c = starting
    #         cnt = 0
    #         while r < self.board.board_size and c >= 0:
    #             if self.board.get(Point(row=r, col=c))==self.prev_player():
    #                 cnt += 1
    #                 if cnt == 5:
    #                     return True
    #             else:
    #                 cnt = 0
    #             r += 1
    #             c -= 1
        
    #     return False    

        # def get_direction(self, direction):
    #     # 0 →
    #     # 1 ←
    #     # 2 ↓
    #     # 3 ↑
    #     # 4 ↘
    #     # 5 ↖
    #     # 6 ↙
    #     # 7 ↗
    #     list_dx = [1, -1, 0,  0, 1, -1,  1, -1]
    #     list_dy = [0,  0, 1, -1, 1, -1, -1,  1]
    #     return list_dx[direction], list_dy[direction]

    # # 새로 착수한 점을 기준으로 n목을 확인
    # def get_stone_count(self, move, direction, player):
    #     x1, y1 = move
    #     cnt = 1
    #     for i in range(2):
    #         dx, dy = self.get_drection(direction)
    #         x, y = x1, y1
    #         while 0 <= x < self.board.board_size and 0 <= y < self.board.board_size:
    #             x, y = x + dx, y + dy
    #             if self.board.get(Point(row=x, col=y))==player:
    #                 cnt += 1
    #             else:
    #                 break
    #     return cnt

    # def is_double_three(self, move):
    #     if self.next_player == Player.white:
    #         return False

    #     next_state = self.apply_move(move)
    #     cnt = 0
    #     for i in range(4):
    #         if next_state.is_open_three(move, i):
    #             cnt += 1
    #             if cnt >= 2:
    #                 return True
    #     return False

    # def is_open_three(self, move, direction):
    #     for i in range(2):
    #         dx, dy = self.get_direction(direction * 2 + i)
    #         point = Point(move.point.row + dx, move.point.col + dy)
    #         if not self.board.is_empty(point):
    #             continue
    #         next_state = self.apply_move(point)
    #         if next_state.is_open_four(move, direction):
    #             return True
    #     return False
             
    # def is_double_four(self, move):
    #     if self.next_player == Player.white:
    #         return False
        
    #     next_state = self.apply_move(move)
    #     cnt = 0
    #     for i in range(4):
    #         if next_state.is_open_four(move, i) == 2:
    #             cnt += 2
    #         elif next_state.is_four(move, i):
    #             cnt += 1
    #         if cnt >= 2:
    #             return True
    #     return False

    # def is_open_four(self, move, direction):
    #     if self.is_five(move):
    #         return False
    #     for i in range(2):
    #         dx, dy = self.get_direction(direction * 2 + 1)
    #         point = Point(move.point.row + dx, move.point.col + dy)
    #         if not self.board.is_empty(point):
    #             continue
    #         if self.is_five(move, direction):
    #             cnt += 1
    #     if cnt == 2:

    #     pass
    
    # # 4 (4목이 아님)
    # def is_four(self, move):


    # # 오목
    # def is_five(self, move, direction=None):
    #     if direction:
    #         return self.get_stone_count(move, direction)
    #     for i in range(4):
    #         if self.get_stone_count(move, i) == 5:
    #             return True
    #     return False

    # # 장목
    # def is_six(self, move):
    #     for i in range(4):
    #         if self.get_stone_count(move, i) > 5:
    #             return True
    #     return False
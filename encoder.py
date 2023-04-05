import numpy as np
import torch

from board import Point
from utils import print_board
# plane 0: empty points
# plane 1: my points
# plane 2: enemy's points
# plane 3: forbidden moves (if player is White, the plane is empty)

class Encoder():
    def __init__(self, board_size):
        self.num_planes = 4
        self.board_size = board_size

    def encode_board(self, game_state):
        # print("ENCODE BOARD")
        # print("game_state.board")
        # print_board(game_state.board)
        board_tensor = np.zeros(self.shape())

        for row in range(self.board_size):
            for col in range(self.board_size):
                p = Point(row=row, col=col)
                stone = game_state.board.get(p)
                if stone==0:
                    board_tensor[0][row][col] = 1
                elif stone==game_state.next_player:
                    board_tensor[1][row][col] = 1
                else:
                    board_tensor[2][row][col] = 1
        # print('before last plane')
        # print_board(game_state.board)
        for move in game_state.forbidden_moves():
            board_tensor[3][move.row][move.col] = 1
        # print('after last plane')
        # print_board(game_state.board)
        return torch.tensor(board_tensor, dtype=torch.float)

    def encode_move(self, point):
        return self.board_size * point.row + point.col

    def decode_move_index(self, index):
        row = index // self.board_size
        col = index % self.board_size
        return Point(row=row, col=col)
    
    def shape(self):
        return self.num_planes, self.board_size, self.board_size

    def num_moves(self):
        return self.board_size * self.board_size
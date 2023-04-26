import numpy as np
import torch

from board import Point
from player import Player

# plane 0: empty points
# plane 1: my stones
# plane 2: enemy's stones
# plane 3: forbidden moves (if player is White, this plane is empty)

class Encoder():
    def __init__(self, board_size):
        self.num_planes = 4
        self.board_size = board_size

    def encode_board(self, game_state):
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
        if game_state.next_player == Player.black:
            for move in game_state.forbidden_moves:
                board_tensor[3][move.row][move.col] = 1
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
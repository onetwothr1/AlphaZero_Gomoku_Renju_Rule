import numpy as np
import torch

from encoders.base import Encoder
from board import Point

# planes description
# 0 empty points
# 1 my points
# 2 enemy's points

class ThreePlaneEncoder(Encoder):
    def __init__(self, board_size):
        self.num_planes = 3
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
        return torch.tensor(board_tensor, dtype=torch.float)

    def encode_point(self, point):
        return self.board_size * point.row + point.col

    def decode_point_index(self, index):
        row = index // self.board_size
        col = index % self.board_size
        return Point(row=row, col=col)
    
    def shape(self):
        return self.num_planes, self.board_size, self.board_size

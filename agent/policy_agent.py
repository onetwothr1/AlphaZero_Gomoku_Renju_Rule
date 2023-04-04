import numpy as np

from agent.base import Agent
from board import NoPossibleMove

class PolicyAgent(Agent):
    def __init__(self, model, encoder):
        self.model = model
        self.encoder = encoder
    
    def select_move(self, game_state):
        board_tensor = self.encoder.encode_board(game_state)
        move_probs = self.model(board_tensor)
        
        num_moves = game_state.board.board_size ** 2
        candidates = np.arange(num_moves)
        ranked_moves = np.random.choice(
            candidates, num_moves, replace=False, p = move_probs
        )

        for point_idx in ranked_moves:
            move = self.encoder.decode_move_index(point_idx)
            if game_state.is_valid_move(move):
                if self.collector is not None:
                    self.collector.record_decision(
                        state = board_tensor,
                        action = point_idx
                    )
                return move
        return NoPossibleMove()
    
    def set_collector(self, collector):
        self.collector = collector
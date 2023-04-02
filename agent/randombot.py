import random
from agent.base import Agent
from board import NoPossibleMove


__all__ = ['RandomBot']


class RandomBot(Agent):
    def select_move(self, game_state):
        candidate = game_state.legal_moves()
        if candidate:
            return random.choice(candidate)
        return NoPossibleMove()
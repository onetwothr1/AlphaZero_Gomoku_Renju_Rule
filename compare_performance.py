import argparse
from tqdm import tqdm
import torch
from scipy.stats import binom_test
from player import Player
from self_play import simulate_game
from agent import AlphaZeroAgent

def performance_comparison(agent1, agent2, board_size, winning_threshold=None, pvalue_threshold=None):
    '''
    agent1: elevated agent
    agent2: previous agent
    '''
    num_games = 100
    agent1_win = 0
    agent2_win = 0
    for i in tqdm(range(num_games)):
        if i % 2 ==0:
            winner = simulate_game(agent1, agent2, board_size)
            if winner == Player.black:
                agent1_win += 1
            else:
                agent2_win += 1
        else:
            winner = simulate_game(agent2, agent1, board_size)
            if winner == Player.black:
                agent2_win += 1
            else:
                agent1_win += 1

    p_val = binom_test(agent1_win, 100, 0.5)
    print('%d wins out of %d' %(agent1_win, num_games))
    print('p-value %.5f' %(p_val))

    if winning_threshold:
        return agent1_win >= winning_threshold
    if pvalue_threshold:
        return p_val <= pvalue_threshold
        
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1', type=str)
    parser.add_argument('--model2', type=str)
    parser.add_argument('--board-size', '-b', type=int, default=9)
    parser.add_argument('--winning-threshold', '-win', type=int, default=60)
    parser.add_argument('--pvalue-threshold', '-pvalue', type=float)
    args = parser.parse_args()

    agent1 = AlphaZeroAgent()
    agent2 = AlphaZeroAgent()
    agent1.load_state_dict(torch.load(args.model1))
    agent2.load_state_dict(torch.load(args.mpdel2))
    performance_comparison(agent1, agent2, args.board_size)

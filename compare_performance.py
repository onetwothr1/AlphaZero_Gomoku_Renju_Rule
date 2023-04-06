import argparse
from tqdm import tqdm
import torch
from scipy.stats import binomtest

from player import Player
from self_play import simulate_game
from agent import *
from net import *
from encoder import Encoder
from utils import *

def performance_comparison(agent1, agent2, board_size, num_games=100, winning_threshold=None, pvalue_threshold=None, verbose=False):
    '''
    agent1: elevated agent
    agent2: previous agent
    '''
    agent1_win = 0
    agent2_win = 0

    agent1_avg_depth_list = []
    agent1_max_depth_list = []
    agent2_avg_depth_list = []
    agent2_max_depth_list = []

    for i in tqdm(range(num_games)):
        if verbose:
            print("////////  %dth Game  ////////" %(i+1))
        if i % 2 ==0:
            if verbose:
                print("Black: agent1, White: agent2")
            winner = simulate_game(agent1, agent2, board_size, verbose)
            if winner == Player.black:
                agent1_win += 1
            else:
                agent2_win += 1
        else:
            if verbose:
                print("Black: agent2, White: agent1")
            winner = simulate_game(agent2, agent1, board_size, verbose)
            if winner == Player.black:
                agent2_win += 1
            else:
                agent1_win += 1
        
        # statistics on tree-depth
        if verbose:
            print()
            print_tree_depth_statistics("Agent 1",
                                        agent1.avg_depth_list,
                                        agent1.max_depth_list,
                                        "Agent 2",
                                        agent2.avg_depth_list,
                                        agent2.max_depth_list)
        agent1_avg_depth_list += agent1.avg_depth_list
        agent1_max_depth_list += agent1.max_depth_list
        agent2_avg_depth_list += agent2.avg_depth_list
        agent2_max_depth_list += agent2.max_depth_list

    p_val = binomtest(agent1_win, num_games, 0.5)
    print("\nComparison finished.")
    print('%d wins out of %d' %(agent1_win, num_games))
    print('p-value %f' %(p_val))

    print()
    print_tree_depth_statistics("Agent 1",
                                agent1_avg_depth_list,
                                agent1_max_depth_list,
                                "Agent 2",
                                agent2_avg_depth_list,
                                agent2_max_depth_list)

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

    board_size = 9
    net = AlphaZeroNet(board_size)
    net.load_state_dict(torch.load('models/alphazero 0.pt'))
    encoder = Encoder(board_size)
    agent1 = AlphaZeroAgent(net, encoder, rounds_per_move=100,
                            c=0.6, is_self_play=False,
                            dirichlet_noise_intensity=0.2,
                            dirichlet_alpha=5, verbose=0)
    agent2 = RandomBot()
    performance_comparison(agent1, agent2, board_size, num_games=20, verbose=True)

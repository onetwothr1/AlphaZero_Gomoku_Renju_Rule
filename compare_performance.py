import argparse
from tqdm import tqdm
import torch
from scipy.stats import binomtest

from self_play import simulate_game
from agent import *
from net.alphazero_net import AlphaZeroNet
from encoder import Encoder
from player import Player
from utils import *

def performance_comparison(agent1, agent2, board_size, num_games=100, winning_threshold=None, pvalue_threshold=None, verbose=False):
    '''
    agent1: elevated agent
    agent2: previous agent
    '''
    agent1_win = 0
    agent1_black_win = 0
    agent2_win = 0
    agent2_black_win = 0

    agent1_avg_depth_list = []
    agent1_max_depth_list = []
    agent2_avg_depth_list = []
    agent2_max_depth_list = []

    for i in tqdm(range(num_games)):
        if verbose:
            print("\n\n////////  %dth Game  ////////" %(i+1))
        if i % 2 ==0:
            if verbose:
                print("Black: %s, White: %s" %(agent1.name, agent2.name))
            winner = simulate_game(agent1, agent2, board_size, verbose)
            if winner == Player.black:
                agent1_win += 1
                agent1_black_win += 1
            else:
                agent2_win += 1
        else:
            if verbose:
                print("Black: %s, White: %s" %(agent2.name, agent1.name))
            winner = simulate_game(agent2, agent1, board_size, verbose)
            if winner == Player.black:
                agent2_win += 1
                agent2_black_win += 1
            else:
                agent1_win += 1
        
        print("\n%s vs %s => %d : %d" %(agent1.name, agent2.name, agent1_win, agent2_win))
        print("playing first: %s vs %s => %d : %d" %(agent1.name, agent2.name, agent1_black_win, agent2_black_win))
        print("playing second: %s vs %s => %d : %d" %(agent1.name, agent2.name, agent1_win - agent1_black_win, agent2_win - agent2_black_win))
        
        # statistics on tree-depth
        if verbose:
            print()
            print_tree_depth_statistics(agent1.name,
                                        agent1.avg_depth_list,
                                        agent1.max_depth_list,
                                        agent2.name,
                                        agent2.avg_depth_list,
                                        agent2.max_depth_list)
        agent1_avg_depth_list += agent1.avg_depth_list
        agent1_max_depth_list += agent1.max_depth_list
        agent2_avg_depth_list += agent2.avg_depth_list
        agent2_max_depth_list += agent2.max_depth_list

    p_val = binomtest(agent1_win, agent1_win + agent2_win, 0.5).pvalue
    print('\n\n//////////////////////////////')
    print("<Comparison Finished>")
    print("\n%s vs %s => %d : %d" %(agent1.name, agent2.name, agent1_win, agent2_win))
    print('p-value %f' %(p_val))
    print("playing first: %s vs %s => %d : %d" %(agent1.name, agent2.name, agent1_black_win, agent2_black_win))
    print("playing second: %s vs %s => %d : %d" %(agent1.name, agent2.name, agent1_win - agent1_black_win, agent2_win - agent2_black_win))
        
    print_tree_depth_statistics(agent1.name,
                                agent1_avg_depth_list,
                                agent1_max_depth_list,
                                agent2.name,
                                agent2_avg_depth_list,
                                agent2_max_depth_list)

    if winning_threshold:
        return agent1_win >= winning_threshold
    if pvalue_threshold:
        return p_val <= pvalue_threshold
        
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1', '-m1', type=str)
    parser.add_argument('--model2', '-m2', type=str)
    parser.add_argument('--board-size', '-b', type=int, default=9)
    parser.add_argument('--num_games', '-n', type=int, default=100)
    parser.add_argument('--winning-threshold', '-win', type=int, default=60)
    parser.add_argument('--pvalue-threshold', '-pvalue', type=float, default=0.05)
    parser.add_argument('--verbose', type=int, default=0)  # 0: none, 1: show play, 2: + progress bar, 3: + thee-depth, 4: + candidate moves
    parser.add_argument('--use-model-name', action='store_true') # with this argument, use agents' own name. else, use names 'agent1' and 'agent2'.
    args = parser.parse_args()

    board_size = 9
    net1 = AlphaZeroNet(board_size)
    net1.load_model(args.model1)
    net2 = AlphaZeroNet(board_size)
    net2.load_model(args.model2)
    encoder = Encoder(board_size)
    agent1 = AlphaZeroAgent(net1, encoder, rounds_per_move=400,
                            c=2.5, is_self_play=False, 
                            # dirichlet_noise_intensity= 0.25,
                            # dirichlet_alpha=0.5,
                            verbose=max(args.verbose-1,0),
                            name=get_model_name(args.model1) if args.use_model_name else 'Agent1')
    agent2 = AlphaZeroAgent(net2, encoder, rounds_per_move=400,
                            c=2.5, is_self_play=False, 
                            # dirichlet_noise_intensity= 0.25,
                            # dirichlet_alpha=0.5,
                            verbose=max(args.verbose-1,0),
                            name=get_model_name(args.model2) if args.use_model_name else 'Agent2')
    # set_stone_color()
    performance_comparison(agent1, agent2, board_size, num_games=args.num_games, verbose=True)

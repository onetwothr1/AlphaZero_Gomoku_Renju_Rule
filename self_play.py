import argparse
from tqdm import tqdm
import time

from experience import *
from agent import *
from board import GameState
from net.alphazero_net import AlphaZeroNet
from encoder import Encoder
from player import Player
from utils import *

def simulate_game(black_player, white_player, board_size, verbose=False):
    game = GameState.new_game(board_size)
    agents = {
        Player.black: black_player,
        Player.white: white_player
    }
    move = None
    while not game.is_over():
        if verbose:
            print('----------------------------')
            print_move(game.prev_player(), move)
            print_board(game.board)

        move = agents[game.next_player].select_move(game)
        game = game.apply_move(move)
    
    if verbose:
        print('----------------------------')
        print_move(game.prev_player(), move)
        print_board(game.board)

        if game.winner:
            if game.win_by_forcing_forbidden_move:
                print_winner(game.winner, game.win_by_forcing_forbidden_move)
            print_winner(game.winner)
        else:
            print_board_is_full()

    return game.winner


def self_play_simulation(agent1, agent2, num_games, save_path, board_size, reward_decay, verbose=False):
    collector1 = ExperienceCollector(reward_decay=reward_decay)
    collector2 = ExperienceCollector(reward_decay=reward_decay)
    agent1.set_collector(collector1)
    agent2.set_collector(collector2)

    i = 0
    while i < num_games: 
        collector1.begin_episode()
        collector2.begin_episode()
        if verbose:
            print("\n////////  Start Game %d  ////////" %(i+1))
        game_record = simulate_game(agent1, agent2, board_size, verbose)
        if game_record == Player.black:
            collector1.complete_episode(reward=1)
            collector2.complete_episode(reward=-1)
            i += 1
            yield # for using tqdm
        elif game_record == Player.white:
            collector1.complete_episode(reward=-1)
            collector2.complete_episode(reward=1)
            i += 1
            yield # for using tqdm
        else:
            pass

    experience = combine_experience(collector1, collector2)
    experience.save_experience(save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-games', '-n', type=int, default=1000)
    parser.add_argument('--model',  '-m')
    parser.add_argument('--file-num', type=int, default=None)
    parser.add_argument('--board-size', '-b', type=int, default=9)
    parser.add_argument('--num-rollout-per-move', '-rollout', type=int, default=1000)
    parser.add_argument('--c', '-c', type=float)
    parser.add_argument('--noise-intensity', type=float)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--reward-decay', type=float, default=0.92)
    parser.add_argument('--verbose', type=int, default=0) # 0: none, 1: show play, 2: + progress bar, 3: + thee-depth, 4: + candidate moves
    args = parser.parse_args()

    net = AlphaZeroNet(args.board_size)
    net.load_state_dict(torch.load(args.model, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    encoder = Encoder(args.board_size)
    agent1 = alphazero_agent.AlphaZeroAgent(net, encoder, args.num_rollout_per_move,
                                            c=args.c, is_self_play=True,
                                            dirichlet_noise_intensity=args.noise_intensity,
                                            dirichlet_alpha=args.alpha, verbose=max(args.verbose-1,0))
    agent2 = alphazero_agent.AlphaZeroAgent(net, encoder, args.num_rollout_per_move,
                                            c=args.c, is_self_play=True,
                                            dirichlet_noise_intensity=args.noise_intensity,
                                            dirichlet_alpha=args.alpha, verbose=max(args.verbose-1,0))
    
    start = time.time()
    # it makes while-loop be able to use 'tqdm' progress bar
    for _ in tqdm(self_play_simulation(agent1, agent2, args.num_games, 
                                       save_path(args.model, args.num_games, args.file_num), 
                                       args.board_size, args.reward_decay, args.verbose)): pass
    time_elapsed = time.time() - start

    # save self-play experience spec
    with open('experience/%s self-play %d.txt' %(save_path(args.model, args.num_games, args.file_num, True)), 'w') as file:
        file.write("rollout per move %d" %args.num_rollout_per_move)
        file.write("\nc %f" %args.c)
        file.write("\ndirichlet noise intensity %f" %args.noise_intensity)
        file.write("\ndirichlet alpha %f" %args.alpha)
        file.write("\ntime elapsed %f" %time_elapsed)

    print('successfully saved self-play experience')
    
if __name__=='__main__':
    main()
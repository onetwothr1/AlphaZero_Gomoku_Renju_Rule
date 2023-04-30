import argparse
from tqdm import tqdm
import time

from experience import *
from alphazero_agent import AlphaZeroAgent
from alphazero_net import AlphaZeroNet
from game import GameState
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
            print_move(game.prev_player(), move, agents[game.prev_player()].name if game.prev_player() else None)
            print_board(game)

        move = agents[game.next_player].select_move(game)
        game = game.apply_move(move)
    
    if verbose:
        print('----------------------------')
        print_move(game.prev_player(), move, agents[game.prev_player()].name)
        print_board(game)

        if game.winner:
            print_winner(game.winner, game.win_by_forcing_forbidden_move)
        elif game.full_board:
            print_board_is_full()
        else:
            print_no_one_can_win()

    return game.winner


def self_play_simulation(agent1, agent2, num_games, save_path, board_size, verbose=False):
    collector1 = ExperienceCollector()
    collector2 = ExperienceCollector()
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
            collector1.complete_episode(reward=0)
            collector2.complete_episode(reward=0)
            i += 1
            yield # for using tqdm

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
    parser.add_argument('--verbose', type=int, default=0) # 0: none, 1: show play, 2: show agent's internal info
    args = parser.parse_args()

    net = AlphaZeroNet(args.board_size)
    net.load_model(args.model)
    encoder = Encoder(args.board_size)
    agent1 = AlphaZeroAgent(net, encoder, args.num_rollout_per_move,
                                            c=args.c, is_self_play=True,
                                            dirichlet_noise_intensity=args.noise_intensity,
                                            dirichlet_alpha=args.alpha, 
                                            verbose= args.verbose>=2)
    agent2 = AlphaZeroAgent(net, encoder, args.num_rollout_per_move,
                                            c=args.c, is_self_play=True,
                                            dirichlet_noise_intensity=args.noise_intensity,
                                            dirichlet_alpha=args.alpha, 
                                            verbose= args.verbose>=2)
    
    start = time.time()
    # below code makes while-loop be able to use 'tqdm' progress bar
    for _ in tqdm(self_play_simulation(agent1, agent2, args.num_games, 
                                       experience_save_path(args.model, args.num_games, args.file_num), 
                                       args.board_size, args.verbose), 
                total=args.num_games): pass
    time_elapsed = time.time() - start

    # save self-play experience spec
    with open(experience_save_path(args.model, args.num_games, args.file_num, '.txt'), 'w') as file:
        file.write("rollout per move %d" %args.num_rollout_per_move)
        file.write("\nc %f" %args.c)
        file.write("\ndirichlet noise intensity %f" %args.noise_intensity)
        file.write("\ndirichlet alpha %f" %args.alpha)
        file.write("\ntime elapsed %ds" %time_elapsed)

    print('successfully saved self-play experience')
    
if __name__=='__main__':
    main()
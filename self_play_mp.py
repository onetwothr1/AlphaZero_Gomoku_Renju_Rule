import argparse
from tqdm import tqdm
import time
from multiprocessing import Process

from experience import *
from agent import *
from board import GameState
from net.alphazero_net import AlphaZeroNet
from encoder import Encoder
from player import Player
from utils import get_model_name, save_path


def simulate_game(black_player, white_player, board_size):
    game = GameState.new_game(board_size)
    agents = {
        Player.black: black_player,
        Player.white: white_player
    }
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        game = game.apply_move(next_move)
    return game.winner

def self_play_simulation(agent1, agent2, num_games, save_path, board_size):
    print('start!')
    collector1 = ExperienceCollector()
    collector2 = ExperienceCollector()
    agent1.set_collector(collector1)
    agent2.set_collector(collector2)

    i = 0
    while i < num_games: 
        collector1.begin_episode()
        collector2.begin_episode()
        game_record = simulate_game(agent1, agent2, board_size)
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
    print('saved')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-games', '-n', type=int, default=1000)
    parser.add_argument('--model',  '-m')
    parser.add_argument('--board-size', '-b', type=int, default=9)
    parser.add_argument('--num-rollout-per-move', '-rollout', type=int, default=1000)
    parser.add_argument('--c', '-c', type=float)
    parser.add_argument('--noise-intensity', type=float)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--verbose', type=int, default=0)
    args = parser.parse_args()

    net = AlphaZeroNet(args.board_size)
    net.load_state_dict(torch.load(args.model))
    encoder = Encoder(args.board_size)
    agent1 = alphazero_agent.AlphaZeroAgent(net, encoder, args.num_rollout_per_move,
                                            c=args.c, is_self_play=True,
                                            dirichlet_noise_intensity=args.noise_intensity,
                                            dirichlet_alpha=args.alpha, verbose=args.verbose)
    agent2 = alphazero_agent.AlphaZeroAgent(net, encoder, args.num_rollout_per_move,
                                            c=args.c, is_self_play=True,
                                            dirichlet_noise_intensity=args.noise_intensity,
                                            dirichlet_alpha=args.alpha, verbose=args.verbose)

    # multiprocess
    start = time.time()
    # num_process = multiprocessing.cpu_count()
    num_process = 2
    games_distribution = [args.num_games // num_process] * num_process
    for i in range(args.num_games % num_process):
        games_distribution[i] += 1

    processes = []
    saved_files = []
    for i in range(num_process):
        _save_path = save_path(args.model, args.num_games, i+1)
        print('distributed:', games_distribution[i])
        process = Process(target=self_play_simulation, 
                            args=(agent1, agent2, games_distribution[i], 
                                _save_path, args.board_size))
        processes.append(process)
        saved_files.append(_save_path)
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    print(time.time() - start)

    # concatenate distributed results
    combine_saved_experience(saved_files, save_path(args.model, args.num_games))

    # save self-play experience spec
    with open('experience/%s self-play %d.txt' %(get_model_name(args.model), args.num_games), 'w') as file:
        file.write("rollout per move %d" %args.num_rollout_per_move)
        file.write("\nc %f" %args.c)
        file.write("\ndirichlet noise intensity %f" %args.noise_intensity)
        file.write("\ndirichlet alpha %f" %args.alpha)

    print('successfully saved self-play experience')
    
if __name__=='__main__':
    main()
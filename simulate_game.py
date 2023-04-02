import argparse
from tqdm import tqdm

from experience import *
from agent import *
from board import GameState
from player import Player
from net import Net
from encoders.encoder import ThreePlaneEncoder


BOARD_SIZE = 9

def simulate_game(black_player, white_player):
    game = GameState.new_game(BOARD_SIZE)
    agents = {
        Player.black: black_player,
        Player.white: white_player
    }
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        game = game.apply_move(next_move)
    return game.winner


def experience_simulation(agent1, agent2, num_games, save_path):
    collector1 = ExperienceCollector()
    collector2 = ExperienceCollector()
    agent1.set_collector(collector1)
    agent2.set_collector(collector2)

    for _ in tqdm(range(num_games)):
        collector1.begin_episode()
        collector2.begin_episode()
        game_record = simulate_game(agent1, agent2)
        if game_record == Player.black:
            collector1.complete_episode(reward=1)
            collector2.complete_episode(reward=-1)
        else:
            collector1.complete_episode(reward=-1)
            collector2.complete_episode(reward=1)
    experience = combine_experience(collector1, collector2)
    experience.save_experience(save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-games', '-n', type=int, default=100)
    parser.add_argument('--file-out', '-f')
    args = parser.parse_args()
    
    net = Net(BOARD_SIZE)
    encoder = ThreePlaneEncoder(BOARD_SIZE)
    agent1 = policy_agent.PolicyAgent(net, encoder)
    agent2 = policy_agent.PolicyAgent(net, encoder)

    experience_simulation(agent1, agent2, args.num_games, args.file_out)
    print('successfully saved experience')
    
if __name__=='__main__':
    main()
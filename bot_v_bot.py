import torch
from numpy import average

from agent import *
from net import *
from encoder import Encoder
from player import Player
from board import GameState
from utils import *

def main():
    board_size = 9
    game = GameState.new_game(board_size)
    model = AlphaZeroNet(board_size)
    model.load_state_dict(torch.load('models/alphazero 0.pt'))
    encoder = Encoder(board_size)

    rounds_per_move = 300
    c = 0.6
    noise_intensity = 0.2
    alpha = 5
    verbose = 3 # 0: none, 1: progress bar, 2: + thee-depth 3: + candidate moves
    bots = {
        Player.black: AlphaZeroAgent(model, encoder, rounds_per_move=rounds_per_move, 
                                     c=c, is_self_play=True, 
                                     dirichlet_noise_intensity=noise_intensity, 
                                     dirichlet_alpha=alpha, verbose=verbose),
        Player.white: AlphaZeroAgent(model, encoder, rounds_per_move=rounds_per_move, 
                                     c=c, is_self_play=True, 
                                     dirichlet_noise_intensity=noise_intensity, 
                                     dirichlet_alpha=alpha, verbose=verbose),
    }
    bot_move = None
    print_board(game.board)

    while not game.is_over():
        # clear_screen()
        print('----------------------------')
        print_move(game.prev_player(), bot_move)
        print_board(game.board)

        bot_move = bots[game.next_player].select_move(game)
        game = game.apply_move(bot_move)

    # clear_screen()
    print('----------------------------')
    print_move(game.prev_player(), bot_move)
    print_board(game.board)

    if game.winner:
        if game.win_by_forcing_forbidden_move:
            print_winner(game.winner, game.win_by_forcing_forbidden_move)
        print_winner(game.winner)
    else:
        print_board_is_full()

    if verbose >= 2:
        print()
        print("Black's average tree-search depth: %.2f" %(average(bots[Player.black].avg_depth)))
        print("Black's average of max depth per each move: %d" %(average(bots[Player.black].avg_max_depth)))
        print("White's average tree-search depth: %.2f" %(average(bots[Player.white].avg_depth)))
        print("White's average of max depth per each move: %d" %(average(bots[Player.white].avg_max_depth)))

if __name__ == '__main__':
    main()
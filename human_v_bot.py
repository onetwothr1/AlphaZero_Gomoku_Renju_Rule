import torch

from agent import *
from net.alphazero_net import AlphaZeroNet
from encoder import Encoder
from board import GameState
from player import Player
from utils import *

def main():
    board_size = 9
    game = GameState.new_game(board_size)
    model = AlphaZeroNet(board_size)
    model.load_state_dict(torch.load('models/alphazero 0.pt'))
    encoder = Encoder(board_size)
    bot = AlphaZeroAgent(model, encoder, rounds_per_move=3, 
                        c=0.6, is_self_play=True, 
                        dirichlet_noise_intensity=0.2, 
                        dirichlet_alpha=5, verbose=1)
    move = None
    print_board(game.board)

    while not game.is_over():
        # clear_screen()
        print('----------------------------')
        print_move(game.prev_player(), move)
        print_board(game.board)

        if game.next_player == Player.black:
            human_input = input('-- ')
            move = handle_input(human_input, game, board_size)
            while move is None:
                human_input = input('-- ')
                move = handle_input(human_input, game, board_size)
        else:
            move = bot.select_move(game)
        game = game.apply_move(move)

    clear_screen()
    print('----------------------------')
    print_move(game.prev_player(), move)
    print_board(game.board)

    if game.winner:
        if game.win_by_forcing_forbidden_move:
            print_winner(game.winner, game.win_by_forcing_forbidden_move)
        print_winner(game.winner)
    else:
        print_board_is_full()


if __name__ == '__main__':
    main()
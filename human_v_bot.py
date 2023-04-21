import torch
from IPython.display import clear_output
import os

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
    model.load_model('models/alphazero 2250.pt')
    encoder = Encoder(board_size)
    bot = AlphaZeroAgent(model, encoder, 
                        rounds_per_move=400, c=2, 
                        is_self_play=False)
    move = None
    # set_stone_color()
    print("Do you want to go first (1) or second (2)?")
    human_turn = int(input())

    if human_turn==1:
      turn = {'human' : Player.black}
    elif human_turn==2:
      turn = {'human' : Player.white}
      
    while not game.is_over():
        # clear_screen()
        clear_output(wait=True)
        print('----------------------------')
        print_move(game.prev_player(), move)
        print_board(game.board)

        if game.next_player == turn['human']:
            human_input = input('-- ')
            move = handle_input(human_input, game, board_size)
            while move is None:
                human_input = input('-- ')
                move = handle_input(human_input, game, board_size)
        else:
            move = bot.select_move(game)
        game = game.apply_move(move)

    # clear_screen()
    clear_output(wait=True)
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
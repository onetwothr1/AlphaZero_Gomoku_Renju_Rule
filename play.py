import torch
from IPython.display import clear_output
import os
import argparse
import threading
import time
from queue import Queue

from alphazero_agent import AlphaZeroAgent
from alphazero_net import AlphaZeroNet
from encoder import Encoder
from board import GameState
from player import Player
from utils import *

def main(num_rounds, verbose):
    board_size = 9
    game = GameState.new_game(board_size)
    model = AlphaZeroNet(board_size)
    model.load_model('models/alphazero 3000.pt')
    encoder = Encoder(board_size)
    bot = AlphaZeroAgent(model, encoder, 
                        rounds_per_move=num_rounds, c=2.5,
                        random_exploration=0.2, 
                        is_self_play=False,
                        verbose=verbose)   
    move = None

    set_stone_color()
    print("Do you want to go first (1) or second (2)?")
    human_turn = int(input())
    if human_turn==1:
      turn = {'human': Player.black}
    elif human_turn==2:
      turn = {'human': Player.white}
    player_name = {
        turn['human']: 'Player',
        turn['human'].other: 'AI'
    }
    # clear_screen()
    clear_output(wait=True)
    
    while not game.is_over():
        # clear_screen()
        clear_output(wait=True)
        print('----------------------------')
        print_move(game.prev_player(), move, player_name[game.prev_player()] if game.prev_player() else None)
        print_board(game)
        print()

        if game.next_player == turn['human']:
            human_input = input('Your move: ')
            move = handle_input(human_input, game, board_size)
            while move is None:
                human_input = input('Your move: ')
                move = handle_input(human_input, game, board_size)
        else:
            print("AI is playing..", end="")
            q = Queue()
            thread = threading.Thread(target=bot.select_move, args=(game, q))
            thread.start()
            while thread.is_alive():
                time.sleep(3)
                print(".", end="", flush=True)
            move = q.get()
        game = game.apply_move(move)

    # clear_screen()
    clear_output(wait=True)
    print('----------------------------')
    print_move(game.prev_player(), move, player_name[game.prev_player()] if game.prev_player() else None)
    print_board(game)

    if game.winner:
        if game.win_by_forcing_forbidden_move:
            print_winner(game.winner, game.win_by_forcing_forbidden_move)
        print_winner(game.winner)
    elif game.full_board:
        print_board_is_full()
    else:
        print_no_one_can_win()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-rounds', '-n', type=int, default=200)
    parser.add_argument('--verbose', type=int, default=0)
    args = parser.parse_args()

    main(num_rounds=args.num_rounds,
        verbose=args.verbose)
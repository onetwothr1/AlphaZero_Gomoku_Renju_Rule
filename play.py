import argparse
import threading
from queue import Queue

from alphazero_agent import AlphaZeroAgent
from alphazero_net import AlphaZeroNet
from encoder import Encoder
from game import GameState
from utils import *

def main(verbose):
    set_stone_color()
    turn, player_name = set_first_player()
    search_num = get_num_searches()
    clear_screen()

    board_size = 9
    game = GameState.new_game(board_size)
    model = AlphaZeroNet(board_size)
    model.load_model('models/alphazero 3000.pt')
    encoder = Encoder(board_size)
    bot = AlphaZeroAgent(model, encoder, 
                    rounds_per_move=search_num, c=2.5,
                    random_exploration=0.2, 
                    is_self_play=False,
                    verbose=verbose)   
    move = None
    
    while not game.is_over():
        if not verbose: clear_screen()
        print('----------------------------')
        print_move(game.prev_player(), move, player_name[game.prev_player()] if game.prev_player() else None)
        print_board(game)
        print()

        if game.next_player == turn['human']:
            move = get_human_move(game, board_size)
        else:
            print("AI is playing..", end="")
            q = Queue()
            thread = threading.Thread(target=bot.select_move, args=(game, q))
            thread.start()
            while thread.is_alive():
                print(".", end="", flush=True)
                thread.join(timeout=1.7)
            move = q.get()
        game = game.apply_move(move)

    if not verbose: clear_screen()
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
    parser.add_argument('--verbose', type=int, default=0) # enter 1 to see AI's internal info
    args = parser.parse_args()

    main(verbose=args.verbose)
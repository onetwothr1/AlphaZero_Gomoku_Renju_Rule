import agent
from player import Player
from board import GameState
from utils import *
import time


def main():
    board_size = 9
    game = GameState.new_game(board_size)
    bots = {
        Player.black: agent.randombot.RandomBot(),
        Player.white: agent.randombot.RandomBot(),
    }
    bot_move = None
    print_board(game.board)

    while not game.is_over():
        # time.sleep(0.1)
        clear_screen()
        print('----------------------------')
        print_move(game.prev_player(), bot_move)
        print_board(game.board)

        bot_move = bots[game.next_player].select_move(game)
        game = game.apply_move(bot_move)

    clear_screen()
    print('----------------------------')
    print_move(game.prev_player(), bot_move)
    print_board(game.board)

    if game.winner:
        if game.win_by_forcing_forbidden_move:
            print_winner(game.winner, game.win_by_forcing_forbidden_move)
        print_winner(game.winner)
    else:
        print_board_is_full()

if __name__ == '__main__':
    main()
from player import Player
from board import GameState, Move
from utils import *

def main():
    board_size = 9
    game = GameState.new_game(board_size)
    move = None
    print_board(game.board)

    while not game.is_over():
        clear_screen()
        print('----------------------------')
        print_move(game.prev_player(), move)
        print_board(game.board)

        human_move = input('-- ')
        point = point_from_coords(human_move.strip())
        move = Move(point)
        print_move(game.next_player, move)
        game = game.apply_move(move)

    clear_screen()
    print_board(game.board)

    if game.winner:
        if game.win_by_forcing_forbidden_move:
            print_winner(game.winner, game.win_by_forcing_forbidden_move)
        print_winner(game.winner)
    else:
        print_board_is_full()

if __name__ == '__main__':
    main()
from board import GameState
from utils import *

def main():
    print("black (1) or white (2) ?")
    player = int(input())
    board_size = 9
    game = GameState.new_game(board_size) if player==1 else GameState.new_game_test(board_size)
    move = None
    
    set_stone_color()
    while not game.is_over():
        clear_screen()
        print('----------------------------')
        print_move(game.prev_player(), move)
        print_board(game.board)

        human_input = input('-- ')
        move = handle_input(human_input, game, board_size)
        while move is None:
            human_input = input('-- ')
            move = handle_input(human_input, game, board_size)
        game = game.apply_move_test(move)

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
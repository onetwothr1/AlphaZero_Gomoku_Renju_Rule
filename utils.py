import matplotlib.pyplot as plt
from numpy import average
import platform
import subprocess
import enum

from board import Point, GameState
from player import Player

COLS = 'ABCDEFGHJKLMNOPQRST'

class StoneIcon:
    black = '●'
    white = '○'
    
    def change():
        StoneIcon.black = '○'
        StoneIcon.white = '●'

def set_stone_color():
    print("Which one is black? Type 1 or 2.")
    print("1: %s, 2: %s" %(StoneIcon.black, StoneIcon.white))
    while True:
        answer = input()
        try:
            answer = int(answer)
        except:
            print('Wrong input. Type only one of 1 or 2.')
            continue
        if answer==Player.black.value:
            print('Stone color is now set.\n')
            break
        elif answer==Player.white.value:
            StoneIcon.change()
            print('Stone color is now set.\n')
            break
        else:
            print('Wrong input. Type only one of 1 or 2.')

def print_move(player, move, player_name=None):
    if player is None or move is None:
        return 0
    move_str = '%s%d' % (COLS[move.col], move.row)
    if player_name:
        print('%s (%s) %s' % (player, player_name, move_str))
    else:
        print('%s %s' % (player, move_str))


def print_board(board):
    # for row in range(board.board_size-1, -1, -1):
    #     line = []
    #     for col in range(board.board_size):
    #         stone = board.get(Point(row=row, col=col))
    #         if stone==0:
    #             line.append(' ')
    #         elif stone==Player.black:
    #             line.append(StoneIcon.black)
    #         elif stone==Player.white:
    #             line.append(StoneIcon.white)
    #     print(' %d %s' % (row, ' '.join(line)))
    # print('   ' + ' '.join(COLS[:board.board_size]))
    for row in range(board.board_size-1, -1, -1):
        line = []
        for col in range(board.board_size):
            stone = board.get(Point(row=row, col=col))
            if stone==0:
                line.append('  ')
            elif stone==Player.black:
                line.append(StoneIcon.black)
            elif stone==Player.white:
                line.append(StoneIcon.white)
        print(' %d %s' % (row, ''.join(line)))
    print('   ' + ' '.join(COLS[:board.board_size]))

def handle_input(_input, game: GameState, board_size):
    point = point_from_coords(_input.strip(), board_size)
    if point is None:
        return None
    if not game.is_empty(point):
        print_not_empty()
        return None
    if not game.is_valid_move(point):
        print_invalid_move()
        return None
    return point

def point_from_coords(coords, board_size):
    try:
        col = COLS.index(str.upper(coords[0]))
        row = int(coords[1:])
        if not (0 <= col < board_size and 0 <= row < board_size):
            print_out_of_board()
            return None
        return Point(row=row, col=col)
    except:
        print_wrong_input()
        return None

def coords_from_point(point):
    return '%s%d' % (
        COLS[point.col],
        point.row
    )

def print_winner(winner, win_by_forcing_forbidden_move=False):
    if win_by_forcing_forbidden_move:
        print('\n%s wins by forcing %s to do forbidden move!' %(winner, winner.other))
    else:
        print('\n%s wins!' %(winner))

def print_board_is_full():
    print("\nboard is full. end game.")

def print_wrong_input():
    print("wrong input. try again")

def print_out_of_board():
    print("point is out of board. type a proper point again.")

def print_not_empty():
    print('the point is already occupied. try another point.')

def print_invalid_move():
    print("that move is forbidden. try another move.")

def print_tree_depth_statistics(player1, p1_avg_depths, p1_max_depths, player2, p2_avg_depths, p2_max_depths):
    print("\n<Tree-search depth statistics>")
    print(player1)
    print("- average depth per each rollout: %.2f" %(average(p1_avg_depths)))
    print("- average of max depth per each move: %.2f" %(average(p1_max_depths)))
    print("- max depth in the game: %d" %(max(p1_max_depths)))
    print()
    print(player2)
    print("- average tree-search depth: %.2f" %(average(p2_avg_depths)))
    print("- average of max depth per each move: %.2f" %(average(p2_max_depths)))
    print("- max depth in the game: %d" %(max(p2_max_depths)))

def clear_screen():
    # see https://stackoverflow.com/a/23075152/323316
    if platform.system() == "Windows":
        subprocess.Popen("cls", shell=True).communicate()
    else:  # Linux and Mac
        print(chr(27) + "[2J")

def get_model_name(model_path):
    return model_path.split('/')[-1].split('.')[0]

def save_path(model_path, total_num_game, i_th=None, extension='.pickle'):
    if i_th:
        return 'experience/%s self-play %d %d%s' %(get_model_name(model_path), total_num_game, i_th, extension)
    else:
        return 'experience/%s self-play %d%s' %(get_model_name(model_path), total_num_game, extension)

def save_graph_img(loss, policy_loss, value_loss, save_path):
    plt.plot(loss, label='loss')
    plt.plot(policy_loss, label='policy loss')
    plt.plot(value_loss, label='value loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig(save_path)
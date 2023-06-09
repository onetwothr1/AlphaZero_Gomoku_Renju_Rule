import matplotlib.pyplot as plt
import numpy as np
from numpy import average
import platform
import subprocess
import sys
from IPython.display import clear_output

from game import Point
from player import Player

COLS = 'ABCDEFGHJKLMNOPQRST'

class StoneIcon:
    black = '●'
    white = '○'
    def change():
        StoneIcon.black = '○'
        StoneIcon.white = '●'

def set_stone_color():
    print("Which color is black? Type 1 or 2.")
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
        print('%s(%s): %s' % (player, player_name, move_str))
    else:
        print('%s: %s' % (player, move_str))

def print_board(game_state):
    board_size = game_state.board.board_size
    
    # For most enviroment
    for row in range(board_size-1, -1, -1):
        line = []
        for col in range(board_size):
            if Point(row=row, col=col) in game_state.forbidden_moves:
                line.append('X ')
                continue
            stone = game_state.board.get(Point(row=row, col=col))
            if stone==0:
                line.append('  ')
            elif stone==Player.black:
                line.append(StoneIcon.black)
            elif stone==Player.white:
                line.append(StoneIcon.white)
        print(' %d %s' % (row, ''.join(line)))
    print('   ' + ' '.join(COLS[:board_size]))

    # Use this code in VSCode terminal environment
    # for row in range(board_size-1, -1, -1):
    #     line = []
    #     for col in range(board_size):
    #         if Point(row=row, col=col) in game_state.forbidden_moves:
    #             line.append('X')
    #             continue
    #         stone = game_state.board.get(Point(row=row, col=col))
    #         if stone==0:
    #             line.append(' ')
    #         elif stone==Player.black:
    #             line.append(StoneIcon.black)
    #         elif stone==Player.white:
    #             line.append(StoneIcon.white)
    #     print(' %d %s' % (row, ' '.join(line)))
    # print('   ' + ' '.join(COLS[:board_size]))


def get_human_move(game, board_size):
    while True:
        try:
            human_input = input("Your move: ")
            point = point_from_coords(human_input.strip(), board_size)
            if point is None:
                raise ValueError
            if not game.is_empty(point):
                print_not_empty()
                raise ValueError
            if not game.is_valid_move(point):
                print_fobidden_move()
                raise ValueError
            return point
        except ValueError:
            pass    

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
    print("\nBoard is full. End game.")

def print_no_one_can_win():
    print("\nThe game has reached a state where no one can win anymore.")

def print_wrong_input():
    print("Wrong input. Please enter a valid input.")

def print_out_of_board():
    print("Point is out of board. Type a valid point on the board.")

def print_not_empty():
    print('The point is already occupied. try another point.')

def print_fobidden_move():
    print("That move is forbidden. try another move.")

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
    if 'ipykernel' in sys.modules:
        clear_output(wait=True)
    elif platform.system() == "Windows":
        subprocess.Popen("cls", shell=True).communicate()
    else:  # Linux and Mac
        print(chr(27) + "[2J")

def set_first_player():
    while True:
        try:
            human_turn = int(input("Do you want to go first (1) or second (2)?: "))
            if human_turn==1:
                turn = {'human': Player.black}
            elif human_turn==2:
                turn = {'human': Player.white}
            else:
                raise ValueError
            player_name = {
                turn['human']: 'Player',
                turn['human'].other: 'AI'
            }
            print()
            return turn, player_name
        except ValueError:
            print("Invalind input. please enter 1 or 2.")

def get_num_searches():
    while True:
        try:
            n = int(input("Enter the number of search iteration for AI tree traversal (default is 200): "))
            if n <= 0:
                print("The number must be a positive integer.")
            else:
                return n
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

def get_model_name(model_path):
    return model_path.split('/')[-1].split('.')[0]

def experience_save_path(model_path, total_num_game, i_th=None, extension='.pickle'):
    if i_th:
        return 'experiences/%s self-play %d %d%s' %(get_model_name(model_path), total_num_game, i_th, extension)
    else:
        return 'experiences/%s self-play %d%s' %(get_model_name(model_path), total_num_game, extension)

def save_graph_img(loss, policy_loss, value_loss, save_path):
    plt.plot(loss, label='loss')
    plt.plot(policy_loss, label='policy loss')
    plt.plot(value_loss, label='value loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    init_loss = loss[0]
    init_policy_loss = policy_loss[0]
    init_value_loss = value_loss[0]
    best_loss = min(loss)
    best_policy_loss = min(policy_loss)
    best_value_loss = min(value_loss)
    plt.yticks([init_loss, init_policy_loss, init_value_loss, best_loss, best_policy_loss, best_value_loss])
    plt.axhline(best_loss, color='gray', linestyle='--')
    plt.axhline(best_policy_loss, color='gray', linestyle='--')
    plt.axhline(best_value_loss, color='gray', linestyle='--')
    
    plt.legend(loc='upper right')
    plt.savefig(save_path)

def show_board_img(game_state):
    board = game_state.board
    board_size = board.board_size
    fig, ax = plt.subplots(figsize=(6, 6))
    for y in range(board_size):
        for x in range(board_size):
            rect = plt.Rectangle((x, board_size - 1 - y), 1, 1, facecolor='white', edgecolor='black', linewidth=0.3)
            ax.add_patch(rect)
            
            if board.get(Point(x,y))==Player.black:
                ax.text(y+0.5, board_size-x-0.5, '●', ha='center',va='center', fontsize=25, color='black')
            elif board.get(Point(x,y))==Player.white:
                ax.text(y+0.5, board_size-x-0.5, '○', ha='center',va='center', fontsize=25, color='black')
    for move in game_state.forbidden_moves:
        ax.text(move.col + 0.5, board_size-move.row - 0.5, 'X', ha='center',va='center', fontsize=25, color='black')
    
    ax.set_xticks(range(board_size))
    ax.set_yticks(range(board_size))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    ax.set_xlim(-0.5, board_size)
    ax.set_ylim(0, board_size)
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()

    COLS = 'ABCDEFGHJKLMNOPQRST'
    for i, letter in enumerate(COLS[:board_size]):
        ax.text(i+0.5,board_size + 0.5, letter,ha='center',va='center')
    for i in range(board_size):
        ax.text(-0.5,board_size - 0.5-i,str(i),ha='center',va='center')
    
    # Remove outer border line
    for spine in ax.spines.values():
        spine.set_visible(False)

    # plt.savefig('animation/%s.png' %game_state.turn_cnt)
    plt.show()
    plt.close()

def visualize_policy_distibution(probability_distribution, game_state):
    board = game_state.board
    board_size = board.board_size
    fig, ax = plt.subplots(figsize=(6, 6))

    for y in range(board_size):
        for x in range(board_size):
            prob = probability_distribution[y * board_size +  x]
            normalized_prob = np.sqrt(prob)
            color = (1 - normalized_prob, 1 - normalized_prob, 1)
            rect = plt.Rectangle((x, board_size - 1 - y), 1, 1, facecolor=color, edgecolor='grey', linewidth=0.3)
            ax.add_patch(rect)

            if board.get(Point(x,y))==Player.black:
                ax.text(y+0.5, board_size-x-0.5, '●', ha='center',va='center', fontsize=25, color='black')
            elif board.get(Point(x,y))==Player.white:
                ax.text(y+0.5, board_size-x-0.5, '○', ha='center',va='center', fontsize=25, color='black')
    for move in game_state.forbidden_moves:
        ax.text(move.col + 0.5, board_size-move.row - 0.5, 'X', ha='center',va='center', fontsize=25, color='black')

    ax.set_xticks(range(board_size))
    ax.set_yticks(range(board_size))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    ax.set_xlim(-0.5, board_size)
    ax.set_ylim(0, board_size)
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()

    COLS = 'ABCDEFGHJKLMNOPQRST'
    for i, letter in enumerate(COLS[:board_size]):
        ax.text(i+0.5,board_size + 0.5, letter,ha='center',va='center')
    for i in range(board_size):
        ax.text(-0.5,board_size - 0.5-i,str(i),ha='center',va='center')
    
    # Remove outer border line
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.show()
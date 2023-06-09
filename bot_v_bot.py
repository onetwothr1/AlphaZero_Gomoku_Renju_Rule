from alphazero_agent import AlphaZeroAgent
from alphazero_net import AlphaZeroNet
from encoder import Encoder
from game import GameState
from player import Player
from utils import *
import matplotlib.animation as animation
import matplotlib.pyplot as plt

def main():
    board_size = 9
    game = GameState.new_game(board_size)
    model = AlphaZeroNet(board_size)
    model.load_model('models/alphazero 3000.pt')
    encoder = Encoder(board_size)

    rounds_per_move = 200
    c = 2.5
    noise_intensity = 0.25
    alpha = 0.2
    verbose = False
    bots = {
        Player.black: AlphaZeroAgent(model, encoder, rounds_per_move=rounds_per_move, 
                                     c=2.5, is_self_play=False, 
                                     dirichlet_noise_intensity=noise_intensity, 
                                     dirichlet_alpha=alpha, 
                                    verbose=verbose),
        Player.white: AlphaZeroAgent(model, encoder, rounds_per_move=rounds_per_move, 
                                     c=2.5, is_self_play=False, 
                                     dirichlet_noise_intensity=noise_intensity, 
                                     dirichlet_alpha=alpha, 
                                    verbose=verbose),
    }
    bot_move = None

    # set_stone_color()
    StoneIcon.change()

    while not game.is_over():
        # clear_screen()
        print('----------------------------')
        print_move(game.prev_player(), bot_move, bots[game.prev_player()].name if game.prev_player() else None)
        print_board(game)
        # show_board_img(game)

        bot_move = bots[game.next_player].select_move(game)
        game = game.apply_move(bot_move)

    # clear_screen()
    print('----------------------------')
    print_move(game.prev_player(), bot_move, bots[game.prev_player()].name)
    print_board(game)
    # show_board_img(game)

    if game.winner:
        print_winner(game.winner, game.win_by_forcing_forbidden_move)
    elif game.full_board:
        print_board_is_full()
    else:
        print_no_one_can_win()

    if verbose:
        print()
        print_tree_depth_statistics("Black",
                                    bots[Player.black].avg_depth_list,
                                    bots[Player.black].max_depth_list,
                                    "White",
                                    bots[Player.white].avg_depth_list,
                                    bots[Player.white].max_depth_list)
        

if __name__ == '__main__':
    main()
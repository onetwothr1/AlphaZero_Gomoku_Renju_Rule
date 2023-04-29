import copy
from collections import namedtuple
from renju_rule import Renju_Rule
from player import Player

class Point(namedtuple('Point', 'row col')):
    def __init__(self, row, col):
        pass

class NoPossibleMove():
    def __init__(self):
        pass

class Board():
    def __init__(self, board_size):
        self.board_size = board_size
        self.grid = []
        self.init_grid()

    def init_grid(self):
        for r in range(self.board_size):
            row = []
            for c in range(self.board_size):
                row.append(0)
            self.grid.append(row)

    def is_empty(self, point):
        return self.grid[point.row][point.col]==0

    def get(self, point):
        return self.grid[point.row][point.col]
    
    def place_stone(self, player, point):
        self.grid[point.row][point.col] = player

    def is_on_grid(self, point):
        return 0 <= point.row < self.board_size and 0 <= point.col < self.board_size

    def is_full(self):
        for row in self.grid:
            for point in row:
                if point == 0:
                    return False
        return True


class GameState():
    def __init__(self, board, next_player, previous, move, turn_cnt):
        self.board = board
        self.next_player = next_player
        self.previous_state = previous
        self.last_move = move
        self.rule = Renju_Rule(self.board)
        self.turn_cnt = turn_cnt
        self.winner = None
        self.win_by_forcing_forbidden_move = False
        self.full_board = False

        self.forbidden_moves = self.get_forbidden_moves()

    def apply_move(self, move):
        if isinstance(move, NoPossibleMove):
            self.win_by_forcing_forbidden_move = True
            self.winner = self.prev_player()
            return self
        next_board = copy.deepcopy(self.board)
        next_board.place_stone(self.next_player, move)
        return GameState(next_board, self.next_player.other, self, move, self.turn_cnt+1)

    # one-man play. for checking applied rules.
    def apply_move_test(self, move):
        next_board = copy.deepcopy(self.board)
        next_board.place_stone(self.next_player, move)
        return GameState(next_board, self.next_player, self, move, self.turn_cnt+1)

    def new_game(board_size: int):
        board = Board(board_size)
        return GameState(board, Player.black, None, None, 0)

    # for board_test, playing on white
    def new_game_test(board_size: int):
        board = Board(board_size)
        return GameState(board, Player.white, None, None)

    def is_empty(self, move):
        return self.board.is_empty(move)
    
    def is_valid_move(self, move):
        if self.next_player == Player.black:
            if self.rule.forbidden_point(move.col, move.row, Player.black):
                return False
        return True

    def legal_moves(self):
        moves = []
        for row in range(0, self.board.board_size):
            for col in range(0, self.board.board_size):
                move = Point(row, col)
                if not self.is_empty(move):
                    continue
                if self.is_valid_move(move):
                    moves.append(move)
        return moves
    
    def get_forbidden_moves(self):
        _forbidden_moves = []
        for row in range(0, self.board.board_size):
            for col in range(0, self.board.board_size):
                if not self.board.is_empty(Point(row, col)):
                    continue
                if self.rule.forbidden_point(col, row, Player.black):
                    _forbidden_moves.append(Point(row=row, col=col))
        return _forbidden_moves

    def is_over(self):
        if self.previous_state is None:
            return False
        if self.board.is_full():
            self.winner = None
            self.full_board = True
            return True
        if self.turn_cnt >= 60 and self.check_draw_situation():
            self.winner = None
            return True
        if self.win_by_forcing_forbidden_move:
            self.forbidden_moves = []
            return True
        if self.check_winning():
            self.winner = self.prev_player()
            self.forbidden_moves = []
            return True
        return False

    def check_winning(self):
        return self.rule.is_five_or_more(self.last_move.col, self.last_move.row, self.prev_player())
    
    def prev_player(self):
        if self.previous_state is None:
            return None
        return self.previous_state.next_player

    def check_draw_situation(self):
        board_temp = copy.deepcopy(self.board)
        for row in board_temp:
            for c in row:
                p = Point(row, c)
                if board_temp.is_empty(p):
                    board_temp.place_stone(self.prev_player())
        return GameState.has_five_in_a_row(board_temp, self.prev_player())

    def has_five_in_a_row(board, stone):
        # Check horizontal sequences
        for row in board:
            for i in range(len(row) - 4):
                if all(cell == stone for cell in row[i:i+5]):
                    return True

        # Check vertical sequences
        for col in range(len(board[0])):
            for i in range(len(board) - 4):
                if all(board[row][col] == stone for row in range(i, i+5)):
                    return True

        # Check diagonal sequences (top-left to bottom-right)
        for row in range(len(board) - 4):
            for col in range(len(board[0]) - 4):
                if all(board[row+i][col+i] == stone for i in range(5)):
                    return True

        # Check diagonal sequences (top-right to bottom-left)
        for row in range(len(board) - 4):
            for col in range(4, len(board[0])):
                if all(board[row+i][col-i] == stone for i in range(5)):
                    return True

        return False















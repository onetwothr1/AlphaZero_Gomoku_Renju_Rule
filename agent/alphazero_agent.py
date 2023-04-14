import torch
import numpy as np
from tqdm import tqdm
import random
from agent import Agent
from board import NoPossibleMove
from utils import coords_from_point

class Branch:
    def __init__(self, prior):
        self.prior = prior # prior probability from policy net
        self.initial_value = None # expected value from value net. 
                                  # we can get this value after branching the node.
        self.visit_count = 0
        self.total_value = 0.0

class AlphaZeroTreeNode:
    def __init__(self, state, value, priors, parent, last_move):
        self.state = state
        self.value = value
        self.parent = parent
        self.last_move = last_move
        self.total_visit_count = 1
        self.branches = {}

        for move, p in priors.items():
            if state.is_empty(move) and state.is_valid_move(move):
                self.branches[move] = Branch(p)
        self.children = {} 

    def moves(self):
        return self.branches.keys()

    def add_child(self, move, child_node, value):
        self.children[move] = child_node
        self.branches[move].initial_value = value

    def has_child(self, move):
        return move in self.children

    def get_child(self, move):
        return self.children[move]  

    def expected_value(self, move):
        branch = self.branches[move]
        if branch.visit_count == 0:
            return 0.0
        return branch.total_value / branch.visit_count
    
    def prior(self, move):
        return self.branches[move].prior
    
    def initial_value(self, move):
        return self.branches[move].initial_value
    
    def visit_count(self, move):
        if move in self.branches:
            return self.branches[move].visit_count
        return 0
    
    def record_visit(self, move, value):
        self.total_visit_count += 1
        self.branches[move].visit_count += 1
        self.branches[move].total_value += value

class AlphaZeroAgent(Agent):
    def __init__(self, model, encoder, rounds_per_move=1000, c=2.0, 
                 is_self_play=False, dirichlet_noise_intensity=0.25, 
                 dirichlet_alpha=0.05, verbose=0, name=None):
        self.model = model
        self.encoder = encoder
        self.num_rounds = rounds_per_move
        self.c = c
        self.is_self_play = is_self_play
        self.noise_intensity = dirichlet_noise_intensity
        self.alpha = dirichlet_alpha
        self.verbose = verbose # 0: none, 1: progress bar, 2: + thee-depth 3: + candidate moves
        self.name = name
        self.reward_decay = 0.95
        self.collector = None # used when generating self-play data
        self.avg_depth_list = [] # average of tree-depth in MCTS
        self.max_depth_list = [] # max tree-depth per each move
    
    def select_move(self, game_state):
        # Tree Search
        root = self.create_node(game_state)
        depth_cnt_list = []
        for _ in tqdm(range(self.num_rounds), disable=np.True_):
            # Walking down the tree
            depth_cnt = 1
            node = root
            next_move = self.select_branch(node)
            while node.has_child(next_move):
                node = node.get_child(next_move)
                next_move = self.select_branch(node)
                depth_cnt += 1
            depth_cnt_list.append(depth_cnt)

            if next_move is not None:
                # Expanding the tree
                new_state = node.state.apply_move(next_move)
                if new_state.check_winning(): # win
                    value = 1 # winner gets explicit reward from game result
                elif new_state.board.is_full(): #draw
                    value = 0 # get explicit reward from game result
                else:
                    child_node = self.create_node(
                        new_state, last_move=next_move, parent=node)
                    value = -1 * child_node.value
                move = next_move
            else: 
                # no possible move available except forbidden moves. current player lost.
                value = 1 # winner gets explicit reward from game result
                move = node.last_move
                node = node.parent # winner

            # Back up
            while node is not None:
                node.record_visit(move, value)
                move = node.last_move
                node = node.parent
                value = -1 * self.reward_decay * value
        
        # Statistics on tree-depth
        avg_depth = sum(depth_cnt_list) / len(depth_cnt_list)
        max_depth = max(depth_cnt_list)
        self.avg_depth_list.append(avg_depth)
        self.max_depth_list.append(max_depth)
        if self.verbose >= 2:
            print('average depth: %.2f, max depth: %d' %(avg_depth, max_depth))

        # If only moves left are forbidden moves
        if len(root.moves()) == 0:
            return NoPossibleMove()
        
        # Record on experience collrector
        if self.collector is not None:
            # print("Record on experience")
            root_state_tensor = self.encoder.encode_board(game_state)
            visit_counts = np.array([
                root.visit_count(
                    self.encoder.decode_move_index(idx))
                for idx in range(self.encoder.num_moves())
            ])
            mcts_prob = visit_counts / np.sum(visit_counts)
            self.collector.record_decision(
                root_state_tensor, mcts_prob)

        # Select a move
        if self.verbose >= 3:
            # print candidate moves with there visit count and output of policy net and value net.
            # if a candidate move has never been visited, it can not show the move's value. 
            for top_move in sorted(root.moves(), key=root.visit_count, reverse=True)[:10]:
                print(coords_from_point(top_move),
                      '    visit %3d  prior %.3f  value %s'
                        %(root.visit_count(top_move), root.prior(top_move), 
                          '%.2f'%(root.initial_value(top_move)) if root.initial_value(top_move) else '???'))

        most_visit_move = max(root.moves(), key=root.visit_count)
        max_visit = root.visit_count(most_visit_move)
        max_tie_list = [move for move in root.moves() if root.visit_count(move)==max_visit]
        return random.choice(max_tie_list)

    def set_collector(self, collector):
        self.collector = collector

    def select_branch(self, node):
        if len(node.moves()) == 0:
            return None

        # exploration
        epsilon = 0.1
        if random.random() < epsilon:
            return random.choice(list(node.moves()))
        
        # exploitation
        total_n = node.total_visit_count

        def score_branch(move):
            q = node.expected_value(move)
            p = node.prior(move)
            n = node.visit_count(move)
            return q + self.c * p * np.sqrt(total_n) / (n + 1)
        
        return max(node.moves(), key=score_branch)

    def create_node(self, game_state, last_move=None, parent=None):
        state_tensor = self.encoder.encode_board(game_state)
        log_priors, values = self.model(state_tensor)
        priors = torch.exp(log_priors[0])
        value = values[0][0]    

        if self.is_self_play:
            # add Dirichlet noise for exploration
            priors = priors.detach().numpy()
            priors = ((1 - self.noise_intensity) * priors
                + self.noise_intensity * 
                np.random.dirichlet(self.alpha * np.ones(len(priors))))

        move_priors = {
            self.encoder.decode_move_index(idx): p
            for idx, p in enumerate(priors)
        }                                                      
        new_node = AlphaZeroTreeNode(
            game_state, value,
            move_priors,
            parent, last_move)
        if parent is not None:
            parent.add_child(last_move, new_node, value)
        return new_node
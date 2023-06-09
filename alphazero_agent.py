import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from game import NoPossibleMove
from utils import coords_from_point, visualize_policy_distibution

class Branch:
    def __init__(self, prior):
        self.prior = prior # prior probability from policy net
        self.initial_value = None # expected value from value net. can get this value after branching the node.
        self.visit_count = 0
        self.total_value = 0.0
        self.loss = 0
        self.winning = 0
        self.proactive_defense = 0
        self.depth_list = []

class AlphaZeroTreeNode:
    def __init__(self, state, value, priors, parent, last_move):
        self.state = state
        self.value = value
        self.parent = parent
        self.priors = priors
        self.last_move = last_move
        self.total_visit_count = 1
        self.branches = {}

        for move, p in priors.items():
            if state.is_empty(move) and state.is_valid_move(move):
                self.branches[move] = Branch(p)
        self.children = {} 

    def moves(self):
        return self.branches.keys()

    def valid_move(self, move):
        return move in self.branches

    def add_child(self, move, child_node, value):
        self.children[move] = child_node
        self.branches[move].initial_value = value

    def has_child(self, move):
        return move in self.children

    def get_child(self, move):
        return self.children[move]  

    def get_expected_value(self, move):
        branch = self.branches[move]
        if branch.visit_count == 0:
            return 0.0
        return branch.total_value / branch.visit_count
    
    def get_prior(self, move):
        return self.branches[move].prior
    
    def get_initial_value(self, move):
        return self.branches[move].initial_value

    def get_total_value(self, move):
        return self.branches[move].total_value
    
    def increase_loss(self, move):
        self.branches[move].loss += 1

    def get_loss(self, move):
        if move in self.branches:
            return self.branches[move].loss
        return 0

    def increase_winning(self, move):
        self.branches[move].winning += 1

    def get_winning(self, move):
        if move in self.branches:
            return self.branches[move].winning
        return 0
        
    def increase_proactive_defense(self, move):
        self.branches[move].proactive_defense += 1

    def get_proactive_defense(self, move):
        if move in self.branches:
            return self.branches[move].proactive_defense

    def append_depth_list(self, move, depth):
        self.branches[move].depth_list.append(depth)

    def show_depth_list(self, move):
        plt.plot(self.branches[move].depth_list)
        plt.xlabel('Number of searches')
        plt.ylabel('Depth')
        plt.show()
    
    def get_max_depth(self, move):
        if self.branches[move].depth_list:
            return max(self.branches[move].depth_list)
        return 0
    
    def get_visit_count(self, move):
        if move in self.branches:
            return self.branches[move].visit_count
        return 0
    
    def record_visit(self, move, value):
        self.total_visit_count += 1
        self.branches[move].visit_count += 1
        self.branches[move].total_value += value

class AlphaZeroAgent():
    def __init__(self, model, encoder, rounds_per_move=200, c=2.5, 
                 is_self_play=False, dirichlet_noise_intensity=0.25, 
                 dirichlet_alpha=0.05, random_exploration=0.1,
                 name=None, verbose=False,
                 show_search_depth_graph=False, show_policy_distribution=False):
        self.model = model
        self.encoder = encoder
        self.num_rounds = rounds_per_move
        self.c = c
        self.is_self_play = is_self_play
        self.noise_intensity = dirichlet_noise_intensity
        self.alpha = dirichlet_alpha
        self.exploration_epsilon = random_exploration
        self.name = name
        self.verbose = verbose
        self.show_search_depth_graph = show_search_depth_graph
        self.show_policy_distribution = show_policy_distribution
        self.reward_decay = 0.95
        self.collector = None # used when generating self-play data
        self.avg_depth_list = [] # average of tree-depth in MCTS
        self.max_depth_list = [] # max tree-depth per each move
    
    def select_move(self, game_state, thread_queue=None):
        root = self.create_node(game_state)
        depth_cnt_list = []
        search_cnt = 0
        additional_search = 0
        search_waitlist = [] # used for proactive defense system
        least_turns_to_enemy_win = 100 # represents how many turns left to enemy's winning
        original_c = self.c # used when value of c is temporarily modified
        if self.verbose: print()

        # Tree Search
        while True:
            depth_cnt = 1
            node = root
            if search_waitlist:
                next_move = search_waitlist.pop()
                if root.valid_move(next_move): root.increase_proactive_defense(next_move)
                else: next_move = self.select_branch(node)
            else: next_move = self.select_branch(node)
            root_child_node = next_move

            # Walking down the tree
            while node.has_child(next_move):
                node = node.get_child(next_move)
                next_move = self.select_branch(node)
                depth_cnt += 1
            depth_cnt_list.append(depth_cnt)
            root.append_depth_list(root_child_node, depth_cnt)
            last_move = next_move

            if last_move is not None:
                # Expanding the tree
                new_state = node.state.apply_move(last_move)
                
                if new_state.check_winning(): 
                    # win
                    value = 1 # winner gets explicit reward from game result

                    # My winning
                    if new_state.prev_player()==game_state.next_player:
                        root.increase_winning(root_child_node)
                        if depth_cnt==1:
                            search_waitlist = [last_move]
                        elif depth_cnt==3:
                            if least_turns_to_enemy_win > 2:
                                for _ in range(3): search_waitlist.append(last_move)

                    # Enemy's winning
                    else:
                        root.increase_loss(root_child_node)
                        least_turns_to_enemy_win = depth_cnt
                        # if enemy wins in his n moves,
                        # add enemy's winning moves to search-waitlist
                        n = 4
                        if depth_cnt <= n*2:
                            for _ in range(3): search_waitlist.append(last_move)
                            enemy_winning_node = node.parent
                            for __ in range(int(depth_cnt/2) - 1):
                                for _ in range(2): search_waitlist.append(enemy_winning_node.last_move)
                                if enemy_winning_node.parent is not None and enemy_winning_node.parent.parent is not None:
                                    enemy_winning_node = enemy_winning_node.parent.parent
                elif new_state.board.is_full(): 
                    #draw
                    value = 0 # get explicit reward from game result
                else: 
                    child_node = self.create_node(
                        new_state, last_move=last_move, parent=node)
                    value = -1 * child_node.value # get a value predicted by value-network
                move = last_move
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
                if value==1 and new_state.prev_player()==game_state.next_player.other:
                    # when enemy wins, decrease the impact of reward decay to make the agent focus more on defense.
                    reward_decay = 0.99
                else:
                    reward_decay = self.reward_decay
                value = -1 * reward_decay * value

            # end tree searching
            search_cnt += 1
            if search_cnt >= self.num_rounds:
                # if the most visited move and the second one's visit counts are close
                # or higy propability of defeat is expected,
                # try more search
                if game_state.turn_cnt < 8:
                    break
                if len(root.moves()) < 2:
                    break
                if additional_search >= 3:
                    break
                sorted_moves = sorted(root.moves(), key=root.get_visit_count, reverse=True)
                most_visit_move = sorted_moves[0]
                second_visit_move = sorted_moves[1]
                if root.get_loss(most_visit_move) / root.get_visit_count(most_visit_move) > 0.2:
                    self.c /= 3
                    search_cnt -= self.num_rounds // 2
                    additional_search += 1
                    if self.verbose: print("A high probability of defeat is expected. Do additional search.")
                elif root.get_visit_count(most_visit_move) <= root.get_visit_count(second_visit_move) + 15:                
                    search_cnt -= self.num_rounds // 2
                    additional_search += 1
                    if self.verbose: print("The best and the second best move's visit counts are close. Do additional search.")
                else:
                    break
        self.c = original_c
        
        # If only moves left are forbidden moves
        if len(root.moves()) == 0:
            return NoPossibleMove()
        
        # Select a move
        if self.verbose:
            # print candidate moves with there visit count and internal info.
            # if a candidate move has never been visited, it can not show the move's value. 
            for candidate_move in sorted(root.moves(), key=root.get_visit_count, reverse=True)[:10]:
                print(coords_from_point(candidate_move),
                      '   visit %3d  p %.3f  v %s  exp_v %5.2f  win %3d  loss %3d  defense %3d  depth %d'
                        %(root.get_visit_count(candidate_move),
                          root.get_prior(candidate_move),
                          '%5.2f'%(root.get_initial_value(candidate_move)) if root.get_initial_value(candidate_move) else '???',
                          root.get_expected_value(candidate_move),
                          root.get_winning(candidate_move),
                          root.get_loss(candidate_move),
                          root.get_proactive_defense(candidate_move),
                          root.get_max_depth(candidate_move)
                          )
                    )
        most_visit_move = max(root.moves(), key=root.get_visit_count)
        max_visit = root.get_visit_count(most_visit_move)
        max_tie_list = [move for move in root.moves() if root.get_visit_count(move)==max_visit]
        next_move_selected = random.choice(max_tie_list)

        # Record on experience collector
        if self.collector is not None:
            root_state_tensor = self.encoder.encode_board(game_state)
            # Subtract the expected number of defeats from the number of visits
            visit_counts = np.array([
                root.get_visit_count(self.encoder.decode_move_index(idx))
                - root.get_loss(self.encoder.decode_move_index(idx))
                for idx in range(self.encoder.num_moves())
            ])
            mcts_prob = visit_counts / np.sum(visit_counts)
            self.collector.record_decision(
                root_state_tensor, mcts_prob, -1 * root.get_expected_value(next_move_selected))

        # Statistics on tree-depth
        avg_depth = sum(depth_cnt_list) / len(depth_cnt_list)
        max_depth = max(depth_cnt_list)
        self.avg_depth_list.append(avg_depth)
        self.max_depth_list.append(max_depth)
        if self.verbose: print('average depth: %.2f, max depth: %d' %(avg_depth, max_depth))

        if self.show_search_depth_graph: root.show_depth_list(next_move_selected)
        if self.show_policy_distribution: visualize_policy_distibution(list(root.priors.values()), game_state)
        
        # thread_queue is used when playing with human
        if thread_queue:
          thread_queue.put(next_move_selected)
        
        return next_move_selected

    def select_move_vanilla(self, game_state):
        root = self.create_node(game_state)
        search_cnt = 0

        # Tree Search
        while True:
            node = root
            next_move = self.select_branch(node)

            # Walking down the tree
            while node.has_child(next_move):
                node = node.get_child(next_move)
                next_move = self.select_branch(node)
            last_move = next_move

            if last_move is not None:
                # Expanding the tree
                new_state = node.state.apply_move(last_move)
                
                # win
                if new_state.check_winning(): 
                    value = 1 # winner gets explicit reward from game result
                #draw
                elif new_state.board.is_full(): 
                    value = 0 # get explicit reward from game result
                else: 
                    child_node = self.create_node(
                        new_state, last_move=last_move, parent=node)
                    value = -1 * child_node.value # get a value predicted by value-network
                move = last_move
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

            # end tree searching
            search_cnt += 1
            if search_cnt == self.num_rounds:
                break
        
        # If only moves left are forbidden moves
        if len(root.moves()) == 0:
            return NoPossibleMove()
        
        # Select a move
        most_visit_move = max(root.moves(), key=root.get_visit_count)
        max_visit = root.get_visit_count(most_visit_move)
        max_tie_list = [move for move in root.moves() if root.get_visit_count(move)==max_visit]
        next_move_selected = random.choice(max_tie_list)

        return next_move_selected

    def set_collector(self, collector):
        self.collector = collector

    def select_branch(self, node):
        if len(node.moves()) == 0:
            return None

        # exploration
        if random.random() < self.exploration_epsilon:
            return random.choice(list(node.moves()))
        
        # exploitation
        total_n = node.total_visit_count

        def score_branch(move):
            q = node.get_expected_value(move)
            p = node.get_prior(move)
            n = node.get_visit_count(move)
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
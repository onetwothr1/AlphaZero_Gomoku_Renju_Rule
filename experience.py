import numpy as np
import pickle
import torch
from torch.utils.data import Dataset

def combine_experience(experience1, experience2):
    combined_states = experience1.states + experience2.states
    combined_rewards = experience1.rewards + experience2.rewards
    combined_mcts_probs = experience1.mcts_probs + experience2.mcts_probs

    combined = ExperienceCollector()
    combined.states = combined_states
    combined.rewards = combined_rewards
    combined.mcts_probs = combined_mcts_probs
    return combined

def combine_saved_experience(saved_experiences: list, save_path):
    states_list = []
    rewards_list = []
    mcts_probs_list = []
    for experience in saved_experiences:
        print(experience)
        collector = ExperienceCollector()
        collector.load_experience(experience)
        collector.to_tensor()
        print(len(collector))
        states_list.append(collector.states)
        rewards_list.append(torch.tensor(collector.rewards))
        mcts_probs_list.append(collector.mcts_probs)

    combined = ExperienceCollector()
    combined.states = torch.cat(states_list, dim=0)
    combined.rewards = torch.cat(rewards_list, dim=0)
    combined.mcts_probs = torch.cat(mcts_probs_list, dim=0)
    combined.save_experience(save_path)
    print("%d experiences combined and saved" %len(combined.states))
    
def rotate_augmentaion(experience, board_size):
    rotated_states = [experience.states]
    rotated_mcts_probs = [experience.mcts_probs]
    for i in range(1, 4):
        rotated_states.append(torch.rot90(experience.states , i, [2,3]))

        # convert flat-tensor into a 2D board, rotate it, back to flat.
        mcts_probs_2d = np.reshape(experience.mcts_probs, (len(experience), board_size, board_size))
        rotated = torch.rot90(mcts_probs_2d, i, [1,2])
        flat = np.reshape(rotated, (len(rotated), -1))
        rotated_mcts_probs.append(flat)

    augmented = ExperienceCollector()
    augmented.states = torch.cat(rotated_states, dim=0)
    augmented.mcts_probs = torch.cat(rotated_mcts_probs, dim=0)
    augmented.rewards = experience.rewards.repeat(4)
    return augmented

class ExperienceCollector(Dataset):
    def __init__(self, board_size=9, num_encoded_plane=4, reward_decay=0.92):
        self.states = [] # encoded tensor of board
        self.rewards = [] # 1 or -1
        self.mcts_probs = []
        self.board_size = board_size
        self.num_encoded_plane = num_encoded_plane
        self.decay = reward_decay # reward decay
        
        self.current_episode_states = []
        self.current_episode_mcts_probs = []
    
    def begin_episode(self):
        self.current_episode_states = []
        self.current_episode_mcts_probs = []

    def complete_episode(self, reward):
        num_states = len(self.current_episode_states)
        self.states += self.current_episode_states
        self.rewards += [reward * (self.decay ** i) for i in range(num_states-1, -1, -1)]
        self.mcts_probs += self.current_episode_mcts_probs
        
        self.current_episode_states = []
        self.current_episode_mcts_probs = []
    
    def record_decision(self, state, mcts_prob):
        self.current_episode_states.append(state)
        self.current_episode_mcts_probs.append(mcts_prob)

    def save_experience(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump((self.states, self.rewards, self.mcts_probs), f)

    def load_experience(self, path):
        with open(path, 'rb') as f:
            self.states, self.rewards, self.mcts_probs = pickle.load(f)
    
    def to_tensor(self):
        if len(self) > 0:
            self.states = torch.stack(self.states)
            self.rewards = torch.tensor(self.rewards, dtype=torch.float)
            self.mcts_probs = torch.tensor(self.mcts_probs)
        else:
            self.states = torch.empty(torch.Size(
                          [1, self.num_encoded_plane, self.board_size, self.board_size]))
            self.rewards = torch.tensor(self.rewards, dtype=torch.float)
            self.mcts_probs = torch.empty(torch.Size(
                          [1,self.board_size * self.board_size]))

    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        state = self.states[idx]
        reward = self.rewards[idx]
        mcts_prob = self.mcts_probs[idx]
        return state, reward, mcts_prob
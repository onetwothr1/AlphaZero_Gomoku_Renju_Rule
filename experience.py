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

def combine_saved_experiences(saved_experiences: list, save_path):
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
    
def augmentation(experience, board_size):
    augmented_states = []
    augmented_mcts_probs = []
    for i in range(0, 4):
        # rotate
        rotated_states = torch.rot90(experience.states , i, [2,3])
        mcts_probs_2d = np.reshape(experience.mcts_probs, (len(experience), board_size, board_size))
        rotated_mcts_probs_2d = torch.rot90(mcts_probs_2d, i, [1,2])
        rotated_mcts_probs_flat = np.reshape(rotated_mcts_probs_2d, (len(rotated_mcts_probs_2d), -1)) # back to flat tensor

        # flip horizontally
        flipped_states = torch.flip(rotated_states, dims=[2])
        flipped_mcts_probs_2d = torch.flip(rotated_mcts_probs_2d, dims=[1])
        flipped_mcts_probs_flat = np.reshape(flipped_mcts_probs_2d, (len(flipped_mcts_probs_2d), -1))
        
        augmented_states.append(rotated_states)
        augmented_states.append(flipped_states)
        augmented_mcts_probs.append(rotated_mcts_probs_flat)
        augmented_mcts_probs.append(flipped_mcts_probs_flat)

    augmented = ExperienceCollector()
    augmented.states = torch.cat(augmented_states, dim=0)
    augmented.mcts_probs = torch.cat(augmented_mcts_probs, dim=0)
    augmented.rewards = experience.rewards.repeat(8)
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
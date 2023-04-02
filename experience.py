import numpy as np
import pickle
import torch
from torch.utils.data import Dataset


def combine_experience(collector1, collector2):
    combined_states = collector1.states + collector2.states
    combined_actions = collector1.actions + collector2.actions
    combined_rewards = collector1.rewards + collector2.rewards

    collector = ExperienceCollector()
    collector.states = combined_states
    collector.actions = combined_actions
    collector.rewards = combined_rewards
    return collector


# to track each decision of episode and give them reward (which is given at the end).
class ExperienceCollector(Dataset):
    def __init__(self):
        self.states = [] # board_tensor
        self.actions = [] # point index
        self.rewards = [] # 1 or -1
        
        self.current_episode_states = []
        self.current_episode_actions = []
    
    def begin_episode(self):
        self.current_episode_states = []
        self.current_episode_actions = []

    def complete_episode(self, reward):
        num_states = len(self.current_episode_states)
        self.states += self.current_episode_states
        self.actions += self.current_episode_actions
        self.rewards += [reward for _ in range(num_states)]
        
        self.current_episode_states = []
        self.current_episode_actions = []
    
    def record_decision(self, state, action):
        self.current_episode_states.append(state)
        self.current_episode_actions.append(action)

    def save_experience(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump((self.states, self.actions, self.rewards), f)

    def load_experience(self, path):
        with open(path, 'rb') as f:
            self.states, self.actions, self.rewards = pickle.load(f)
        
        self.states = torch.stack(self.states)
        self.actions = torch.tensor(self.actions, dtype=torch.int32)
        self.rewards = torch.tensor(self.rewards, dtype=torch.float)

    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        state = self.states[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        return state, action, reward

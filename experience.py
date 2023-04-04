import numpy as np
import pickle
import torch
from torch.utils.data import Dataset


def combine_experience(collector1, collector2):
    combined_states = collector1.states + collector2.states
    combined_rewards = collector1.rewards + collector2.rewards
    combined_mcts_probs = collector1.mcts_probs + collector2.mcts_probs

    collector = ExperienceCollector()
    collector.states = combined_states
    collector.rewards = combined_rewards
    collector.mcts_probs = combined_mcts_probs
    return collector

def combine_saved_experience(saved_experiences: list, save_path):
    combined_collector = ExperienceCollector()
    for i in range(len(saved_experiences)):
        collector = ExperienceCollector()
        collector.load_experience(saved_experiences[i])
        combined_collector.states += collector.states
        combined_collector.rewards += collector.rewards
        combined_collector.mcts_probs += collector.mcts_probs
    combined_collector.save_experience(save_path)
    print("%d experiences combined and saved" %len(combined_collector.states))
    

class ExperienceCollector(Dataset):
    def __init__(self):
        self.states = [] # board_tensor
        self.rewards = [] # 1 or -1
        self.mcts_probs = []
        
        self.current_episode_states = []
        self.current_episode_mcts_probs = []
    
    def begin_episode(self):
        self.current_episode_states = []
        self.current_episode_mcts_probs = []

    def complete_episode(self, reward):
        num_states = len(self.current_episode_states)
        self.states += self.current_episode_states
        self.rewards += [reward for _ in range(num_states)]
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
        
        self.states = torch.stack(self.states)
        self.rewards = torch.tensor(self.rewards, dtype=torch.float)
        self.mcts_probs = torch.tensor(self.mcts_probs)

    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        state = self.states[idx]
        reward = self.rewards[idx]
        mcts_prob = self.mcts_probs[idx]
        return state, reward, mcts_prob
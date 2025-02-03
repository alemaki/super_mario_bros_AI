import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from os import path

def save_dqn_model(episode, policy_net, save_dir = "dqn_saved_models"):
    policy_path = path.join(save_dir, f"policy_net_episode_{episode}.pth")
    torch.save(policy_net.state_dict(), policy_path)
    print(f"Models saved at episode {episode}")

def load_dqn_models(episode, policy_net, target_net, save_dir = "dqn_saved_models"):
    policy_path = path.join(save_dir, f"policy_net_episode_{episode}.pth")
    target_path = path.join(save_dir, f"policy_net_episode_{episode}.pth")
    policy_net.load_state_dict(torch.load(policy_path))
    target_net.load_state_dict(torch.load(target_path))
    print(f"Models loaded from episode {episode}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions: int, use_leaky_relu: bool = False, channel_multiplier: int = 1):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32*channel_multiplier, kernel_size=8, stride=4),
            nn.LeakyReLU() if use_leaky_relu else nn.ReLU(),
            nn.Conv2d(32*channel_multiplier, 64*channel_multiplier, kernel_size=4, stride=2),
            nn.LeakyReLU() if use_leaky_relu else nn.ReLU(),
            nn.Conv2d(64*channel_multiplier, 64*channel_multiplier, kernel_size=3, stride=1),
            nn.LeakyReLU() if use_leaky_relu else nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(self._feature_size(input_shape), 512*channel_multiplier),
            nn.LeakyReLU() if use_leaky_relu else nn.ReLU(),
            nn.Linear(512*channel_multiplier, n_actions),
        )

    def _feature_size(self, shape):
        with torch.no_grad():
            return self.conv(torch.zeros(1, *shape)).view(1, -1).size(1)

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x.view(x.size(0), -1))
    

    def train_mario(self, 
                    memory: deque,
                    target_net: "Self",
                    optimizer: optim.Adam,
                    gamma: float,
                    batch_size: float = 64):
        batch = random.sample(memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).to(device)  
        next_states = torch.stack(next_states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.FloatTensor(dones).to(device)

        q_values = self(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = target_net(next_states).max(1)[0]
        targets = rewards + gamma * next_q_values * (1 - dones)

        loss: nn.MSELoss = nn.MSELoss()
        output = loss(q_values, targets)
        optimizer.zero_grad()
        output.backward()
        optimizer.step()
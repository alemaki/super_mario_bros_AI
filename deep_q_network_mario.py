import numpy as np
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym.wrappers import TimeLimit
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import time 
import os



SAVE_DIR = "saved_models"
LEARNING_RATE = 1e-4
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.999
EPSILON_UPDATE = 80
BATCH_SIZE = 64
MEMORY_SIZE = 30000
TARGET_UPDATE = 1000
EPISODE_SAVE = 10
MAX_STEPS = 4000

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def save_models(episode, policy_net, target_net):
    policy_path = os.path.join(SAVE_DIR, f"policy_net_episode_{episode}.pth")
    target_path = os.path.join(SAVE_DIR, f"target_net_episode_{episode}.pth")
    torch.save(policy_net.state_dict(), policy_path)
    torch.save(target_net.state_dict(), target_path)
    print(f"Models saved at episode {episode}")

def load_models(episode, policy_net, target_net):
    policy_path = os.path.join(SAVE_DIR, f"policy_net_episode_{episode}.pth")
    target_path = os.path.join(SAVE_DIR, f"target_net_episode_{episode}.pth")
    policy_net.load_state_dict(torch.load(policy_path))
    target_net.load_state_dict(torch.load(target_path))
    print(f"Models loaded from episode {episode}")

env = gym_super_mario_bros.make('SuperMarioBros-v0', max_episode_steps=MAX_STEPS)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Converts the state to grayscale. Crops, downsamples, normalises.
def preprocess_state(state):
    state = np.mean(state, axis=2, keepdims=False).astype(np.uint8)
    state = state[35:195:2, ::2]
    return state / 255.0  # Normalize

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(self._feature_size(input_shape), 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def _feature_size(self, shape):
        with torch.no_grad():
            return self.conv(torch.zeros(1, *shape)).view(1, -1).size(1)

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x.view(x.size(0), -1))

input_shape = (4, 80, 128)  # 4 stacked frames
n_actions = env.action_space.n

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
policy_net = DQN(input_shape, n_actions).to(device)
target_net = DQN(input_shape, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = deque(maxlen=MEMORY_SIZE)

epsilon = EPSILON_START

for episode in range(10000):
    start_time = time.time()
    state = preprocess_state(env.reset())
    state = np.stack([state] * 4, axis=0)  # Stack frames
    done = False
    total_reward = 0

    for frame in range(MAX_STEPS + 1):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = policy_net(state_tensor).argmax(dim=1).item()

        next_state, reward, done, truncated, _ = env.step(action)
        next_state = preprocess_state(next_state)
        next_state = np.append(state[1:], np.expand_dims(next_state, axis=0), axis=0)

        if done or truncated: 
            break
        
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        # Training
        if len(memory) > BATCH_SIZE:
            batch = random.sample(memory, BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(np.array(states)).to(device)
            actions = torch.LongTensor(actions).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            next_states = torch.FloatTensor(np.array(next_states)).to(device)
            dones = torch.FloatTensor(dones).to(device)

            q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_values = target_net(next_states).max(1)[0]
            targets = rewards + GAMMA * next_q_values * (1 - dones)

            loss = nn.MSELoss()(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        if frame % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if frame % EPSILON_UPDATE == 0:
            epsilon = max(EPSILON_MIN, epsilon*EPSILON_DECAY)

    elapsed_time = time.time() - start_time
    print(f"Episode {episode}, Total Reward: {total_reward}, Time Elapsed: {elapsed_time:.2f} seconds, epsilon {epsilon}")

    if episode % EPISODE_SAVE == 0:
        save_models(episode, policy_net, target_net)


env.close()
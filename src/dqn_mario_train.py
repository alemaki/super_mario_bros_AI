import numpy as np
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import torch
import torch.optim as optim
from collections import deque
import time 
import os
from deep_q_network.deep_q_network import DQN, device, save_dqn_models, load_dqn_models
from utils import preprocess_state

SAVE_DIR = "dqn_saved_models"
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

env = gym_super_mario_bros.make('SuperMarioBros-v0', max_episode_steps=MAX_STEPS)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

input_shape = preprocess_state(env.reset(), device=device).unsqueeze(0).repeat(4, 1, 1).shape  # 4 stacked frames
n_actions = env.action_space.n
policy_net = DQN(input_shape, n_actions).to(device)
target_net = DQN(input_shape, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = deque(maxlen=MEMORY_SIZE)

epsilon = EPSILON_START

for episode in range(10000):
    start_time = time.time()
    state = preprocess_state(env.reset(), device=device)
    state = state.unsqueeze(0).repeat(4, 1, 1)

    done = False
    total_reward = 0

    for frame in range(MAX_STEPS + 1):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = state.unsqueeze(0) # Add batch dimension
            action = policy_net(state_tensor).argmax(dim=1).item()

        next_state, reward, done, truncated, _ = env.step(action)
        next_state = preprocess_state(next_state, device=device)
        next_state = torch.cat((state[1:], next_state.unsqueeze(0)), dim=0)  # Update frame stack

        if done or truncated: 
            break
        
        memory.append((state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done)) # store in RAM for faster sampling. Then transform again.
        state = next_state
        total_reward += reward

        # Training
        if len(memory) > BATCH_SIZE:
            policy_net.train_mario(memory,
                                   target_net,
                                   optimizer,
                                   GAMMA,
                                   BATCH_SIZE)

        if frame % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if frame % EPSILON_UPDATE == 0:
            epsilon = max(EPSILON_MIN, epsilon*EPSILON_DECAY)

    elapsed_time = time.time() - start_time
    print(f"Episode {episode}, Total Reward: {total_reward}, Time Elapsed: {elapsed_time:.2f} seconds, epsilon {epsilon}")

    if episode % EPISODE_SAVE == 0:
        save_dqn_models(episode, policy_net, target_net, SAVE_DIR)


env.close()
import numpy as np
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import torch
import torch.optim as optim
from collections import deque
import time 
import os
from deep_q_network.deep_q_network import DQN, device, save_dqn_model, load_dqn_models
from utils import preprocess_state, record_info_for_episode
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SAVE_DIR = BASE_DIR / "dqn_simple_movement_one_life_bigger_models"
LOG_FILE_NAME = BASE_DIR / "dqn_simple_movement_one_life_bigger_models" / "episodes_log.log"
START_MODEL_EPISODE = 600
LEARNING_RATE = 5e-4
GAMMA = 1.01
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.999
EPSILON_UPDATE = 80
BATCH_SIZE = 128
MEMORY_SIZE = 35000 # about 12 gigs of RAM
TARGET_UPDATE = 3000
EPISODE_SAVE = 100
MAX_STEPS = 6000
ONE_LIFE = True
CHANNEL_MULTIPLIER = 2

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

env = gym_super_mario_bros.make('SuperMarioBros-v0', max_episode_steps=MAX_STEPS)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

input_shape = preprocess_state(env.reset(), device=device).unsqueeze(0).repeat(4, 1, 1).shape  # 4 stacked frames
n_actions = env.action_space.n

policy_net = DQN(input_shape, n_actions, True, CHANNEL_MULTIPLIER).to(device)
target_net = DQN(input_shape, n_actions, True, CHANNEL_MULTIPLIER).to(device)
if START_MODEL_EPISODE != -1:
    load_dqn_models(START_MODEL_EPISODE, policy_net, target_net, SAVE_DIR)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = deque(maxlen=MEMORY_SIZE)

epsilon = EPSILON_START
epsilon = max(EPSILON_MIN, epsilon*(EPSILON_DECAY**(START_MODEL_EPISODE*MAX_STEPS/EPSILON_UPDATE)))

target_update_frames_left = 3000

for episode in range(START_MODEL_EPISODE + 1, 10000):
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

        next_state, reward, done, truncated, info = env.step(action)
        next_state = preprocess_state(next_state, device=device)
        next_state = torch.cat((state[1:], next_state.unsqueeze(0)), dim=0)  # Update frame stack

        if done or truncated: 
            break
        
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        # Training
        if len(memory) > BATCH_SIZE:
            policy_net.train_mario(memory,
                                   target_net,
                                   optimizer,
                                   GAMMA,
                                   BATCH_SIZE)

        target_update_frames_left -= 1
        if target_update_frames_left <= 0:
            target_net.load_state_dict(policy_net.state_dict())
            target_update_frames_left = TARGET_UPDATE

        if frame % EPSILON_UPDATE == 0:
            epsilon = max(EPSILON_MIN, epsilon*EPSILON_DECAY)

        if ONE_LIFE and info['life'] < 2:
            break

    elapsed_time = time.time() - start_time
    print(f"Episode {episode}, Total Reward: {total_reward}, Time Elapsed: {elapsed_time:.2f} seconds, epsilon {epsilon}")

    record_info_for_episode(LOG_FILE_NAME, episode, total_reward, elapsed_time, info)

    if episode % EPISODE_SAVE == 0:
        save_dqn_model(episode, policy_net, SAVE_DIR)


env.close()
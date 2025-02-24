import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from actor_critic.actor_critic import ActorCritic, device
from utils import preprocess_smaller_state, get_reward, record_info_for_worker
from pynput import keyboard
from time import time
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
SAVE_DIR = BASE_DIR / "../models/a2c_simple_movement_new_models"
LOG_FILE_NAME = BASE_DIR / "../models/a2c_simple_movement_new_models" / "episodes_log.log"
LOAD_MODEL_EPISODE = -1
LEARNING_RATE = 1e-3
GAMMA = 0.99
MAX_ENV_STEPS = 6000
MAX_EPISODE_TRAIN = 10000
MODEL_SAVE_EPISODES = 1000
ONE_LIFE = True

env = gym_super_mario_bros.make('SuperMarioBros-v0', max_episode_steps=MAX_ENV_STEPS)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

input_shape = preprocess_smaller_state(env.reset(), device).unsqueeze(0).shape

stop_training = False  # Flag to stop training after the next episode

def on_press(key):
    global stop_training
    if key == keyboard.Key.f1:
        print("F1 pressed. The model will save at the end of the next episode.")
        stop_training = True

listener = keyboard.Listener(on_press=on_press)
listener.start()

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def save_model(model, episode, save_dir=SAVE_DIR):
    model_path = save_dir / f"a2c_model_episode_{episode}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at episode {episode} to {model_path}")

def load_model(model, episode, save_dir=SAVE_DIR):
    model_path = save_dir / f"a2c_model_episode_{episode}.pth"
    model.load_state_dict(torch.load(model_path))
    print(f"Model loaded from episode {episode} from {model_path}")



def train():
    print("Press F1 to stop")
    n_actions = len(SIMPLE_MOVEMENT)
    model = ActorCritic(input_shape, n_actions).to(device)

    if LOAD_MODEL_EPISODE != -1:
        load_model(model, LOAD_MODEL_EPISODE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for episode in range(MAX_EPISODE_TRAIN + 1):
        start = time()
        state = preprocess_smaller_state(env.reset(), device).unsqueeze(0).unsqueeze(0) # two times for grayscale and batch size
        done = False
        log_probs, values, rewards = [], [], []
        
        while not done:
            env.render()
            logits, value = model(state)
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs.squeeze(0), 1).item()  # Remove batch dim thats why squeeze
            log_prob = torch.log(probs.squeeze(0)[action])
            next_state, reward, terminated, truncated, info = env.step(action)

            reward = get_reward(info, reward)
        
            next_state = preprocess_smaller_state(next_state, device).unsqueeze(0).unsqueeze(0)
            done = terminated or truncated
            
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            
            state = next_state

            if info['life'] < 2 and ONE_LIFE:
                break
        
        returns, G = [], 0
        for r in reversed(rewards):
            G = r + GAMMA * G
            returns.insert(0, G)

        returns = torch.tensor(returns).to(device)
        values = torch.cat(values).to(device)
        log_probs = torch.stack(log_probs).to(device)
        
        advantage = returns - values.squeeze()
        actor_loss = -log_probs * advantage.detach()
        critic_loss = advantage.pow(2)
        loss = actor_loss.mean() + critic_loss.mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_reward = sum(rewards)
        estimated_reward = sum(values).item()
        took_time = time() - start

        print(f"Finished episode: {episode}. Time: {took_time:.2f}. Total reward: {total_reward}. Estimated reward: {estimated_reward}")
        record_info_for_worker(LOG_FILE_NAME, episode, 0, total_reward, took_time, info)


        if stop_training:
            print("Stopping training after this episode and saving model...")
            save_model(model, episode)
            break

        if episode % MODEL_SAVE_EPISODES == 0:
            print("saving model...")
            save_model(model, episode)
    
    env.close()
    torch.save(model.state_dict(), "mario_actor_critic.pth")






train()

import torch
import torch.optim as optim
import torch.multiprocessing as mp
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from actor_critic.actor_critic import ActorCritic, device
from utils import preprocess_smaller_state, record_info_for_worker
import os
from pathlib import Path
import signal
import sys
from pynput import keyboard
import time

# Set the multiprocessing start method (only once)
if __name__ == "__main__":
    try:
        mp.set_start_method('spawn')  # Use 'spawn' for multiprocessing
    except RuntimeError:
        print("Multiprocessing context already set.")

# Define the save directory
BASE_DIR = Path(__file__).resolve().parent
SAVE_DIR = BASE_DIR / "a3c_simple_movement_models"
LOG_FILE_NAME = BASE_DIR / "a3c_simple_movement_models" / "episodes_log.log"
LEARNING_RATE = 1e-4
GAMMA = 0.9
ENTROPY_COEF = 0.01
MODEL_SAVE_EPISODES = 100
N_STEPS = 20
MAX_STEPS_ENV = 5000

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def save_model(global_model, episode, save_dir=SAVE_DIR):
    model_path = save_dir / f"global_model_episode_{episode}.pth"
    torch.save(global_model.state_dict(), model_path)
    print(f"Model saved at episode {episode} to {model_path}")

def load_model(global_model, episode, save_dir=SAVE_DIR):
    model_path = save_dir / f"global_model_episode_{episode}.pth"
    global_model.load_state_dict(torch.load(model_path))
    print(f"Model loaded from episode {episode} from {model_path}")

def compute_advantages_and_update(
        global_model: ActorCritic,
        optimizer,
        states,
        actions,
        rewards,
        values,
        done = False):


    states = torch.stack(states).squeeze(dim=1).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    values = torch.stack(values).squeeze().to(device)

    R = 0

    if not done:
        _, next_value = global_model(states[-1].unsqueeze(0))
        R = next_value.item()
    returns = []

    for r in reversed(rewards):
        R = r + GAMMA * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32).to(device)

    advantages = returns - values


    action_probs, _ = global_model(states)
    log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))
    policy_loss = -(log_probs * advantages.detach()).mean()

    value_loss = 0.5 * (returns - values).pow(2).mean()

    entropy = -(action_probs * torch.log(action_probs)).sum(dim=1).mean()
    entropy_loss = -ENTROPY_COEF * entropy

    total_loss = policy_loss + value_loss + entropy_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

def worker(global_model,
           optimizer,
           worker_id,
           env_name,
           n_actions,
           stop_flag):
    
    print(f"Worker {worker_id} started.")

    env = gym_super_mario_bros.make(env_name, max_episode_steps=MAX_STEPS_ENV)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    input_shape = preprocess_smaller_state(env.reset(), device).shape
    local_model = ActorCritic(input_shape, n_actions).to(device)

    with global_model_lock:
        local_model.load_state_dict(global_model.state_dict())

    while not stop_flag.value:
        start_time = time.time()
        state = preprocess_smaller_state(env.reset(), device)
        done = False
        total_reward = 0

        while not done:
            states, actions, rewards, values = [], [], [], []
            for _ in range(N_STEPS):
                state = state.unsqueeze(0).unsqueeze(0) # add batch and channel dimensions.
                action_probs, value = local_model(state)
                action = torch.multinomial(action_probs, 1).item()

                next_state, reward, done, truncated, info = env.step(action)
                next_state = preprocess_smaller_state(next_state, device)
                total_reward += reward

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                values.append(value)

                state = next_state
                if done or truncated:
                    break

            with global_model_save_lock:
                compute_advantages_and_update(global_model,
                                                optimizer,
                                                states,
                                                actions,
                                                rewards,
                                                values,
                                                done)

                local_model.load_state_dict(global_model.state_dict())

                

        elapsed_time = time.time() - start_time

        with global_model_lock:
            global_counter.value += 1
            current_episode = global_counter.value
            record_info_for_worker(LOG_FILE_NAME, current_episode, worker_id, elapsed_time, total_reward, info)
            print(f"Worker {worker_id} finished episode: {current_episode}. Time: {elapsed_time}. Total reward: {total_reward}.")

        if current_episode % MODEL_SAVE_EPISODES == 0:
            with global_model_lock:
                save_model(global_model, current_episode)
    
    print(f"Worker {worker_id} stopped.")

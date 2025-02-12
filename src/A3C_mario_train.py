import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn.functional as F
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from actor_critic.actor_critic import ActorCritic, device
from actor_critic.shared_adam import SharedAdam
from utils import preprocess_smaller_state, record_info_for_worker, get_reward
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

if __name__ == "__main__":
    global_counter = mp.Value('i', -1)
    episode_count_lock = mp.Lock()
    global_model_lock = mp.Lock()
    stop_flag = mp.Value('b', False)

# Define the save directory
BASE_DIR = Path(__file__).resolve().parent
SAVE_DIR = BASE_DIR / "a2c_simple_movement_models"
LOG_FILE_NAME = BASE_DIR / "a2c_simple_movement_models" / "episodes_log.log"
LOAD_MODEL_EPISODE = -1
LEARNING_RATE = 1e-4
GAMMA = 0.99
ENTROPY_COEF = 0.05
MODEL_SAVE_EPISODES = 1000
N_STEPS = 12
ACTION_STEPS = 4
MAX_STEPS_ENV = 5000
MAX_EPISODES = 50000
ONE_LIFE = True

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
        global_model,
        local_model,
        optimizer,
        states,
        actions,
        rewards,
        done,
        global_model_lock):

    # Compute loss on LOCAL model
    states = torch.stack(states).squeeze(dim=1).to(device)
    actions = torch.tensor(actions, dtype=torch.long, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

    action_probs, values = local_model(states)
    values = values.squeeze()

    # Compute returns and advantages (as before)
    R = local_model(states[-1].unsqueeze(0))[1].item()
    returns = []
    for r in reversed(rewards):
        R = r + GAMMA * R
        returns.insert(0, R)
    returns = torch.tensor(returns, device=device)

    advantages = returns - values

    log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))
    policy_loss = -(log_probs * advantages.detach()).mean()
    value_loss = 0.5 * (returns - values).pow(2).mean()
    entropy = -(action_probs * torch.log(action_probs)).sum(dim=1).mean()
    entropy_loss = -ENTROPY_COEF * entropy

    total_loss = policy_loss + value_loss + entropy_loss


    # for name, param in global_model.named_parameters():
    #     if param.grad is not None:
    #         print(f"global_model {name} gradient norm: {param.grad.norm().item()}")
    #     else:
    #         print(f"global_model {name} gradient norm: {None}")

    # for name, param in local_model.named_parameters():
    #     if param.grad is not None:
    #         print(f"local_model {name} gradient norm: {param.grad.norm().item()}")
    #     else:
    #         print(f"local_model {name} gradient norm: {None}")


    optimizer.zero_grad()
    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(local_model.parameters(), 0.5)

    with global_model_lock:
        for local_param, global_param in zip(local_model.parameters(), 
                                           global_model.parameters()):
            if global_param.grad is None: 
                global_param.grad = torch.zeros_like(global_param.data)
            global_param.grad.data.copy_(local_param.grad.data)

        optimizer.step()

        local_model.load_state_dict(global_model.state_dict())
    
    
    # for name, param in global_model.named_parameters():
    #     if param.grad is not None:
    #         print(f"global_model {name} gradient norm: {param.grad.norm().item()}")
    #     else:
    #         print(f"global_model {name} gradient norm: {None}")

    # for name, param in local_model.named_parameters():
    #     if param.grad is not None:
    #         print(f"local_model {name} gradient norm: {param.grad.norm().item()}")
    #     else:
    #         print(f"local_model {name} gradient norm: {None}")

def worker(global_model,
           optimizer,
           worker_id,
           env_name,
           n_actions,
           global_counter,
           episode_count_lock,
           global_model_lock,
           stop_flag):

    env = gym_super_mario_bros.make(env_name, max_episode_steps=MAX_STEPS_ENV)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    input_shape = preprocess_smaller_state(env.reset(), device).shape
    local_model = ActorCritic(input_shape, n_actions, True, 2).to(device)
    local_model.train()

    print(f"Worker {worker_id} started.")

    with global_model_lock:
        local_model.load_state_dict(global_model.state_dict())

    while not stop_flag.value and global_counter.value <= MAX_EPISODES:
        # old_state_dict = {}
        # for key in global_model.state_dict():
        #     old_state_dict[key] = global_model.state_dict()[key].clone()
        start_time = time.time()
        state = preprocess_smaller_state(env.reset(), device)
        done = False
        truncated = False
        total_reward = 0
        estimated_reward = 0
        state = state.unsqueeze(0).unsqueeze(0)
    
        while not done and not truncated:
            #print(f"Worker {worker_id} calls")
            states, actions, rewards = [], [], []
            for _ in range(N_STEPS//ACTION_STEPS):
                env.render()
                with torch.no_grad():
                    action_probs, value = local_model(state)

                #print(action_probs)
                action = torch.multinomial(action_probs, 1).item()

                for _ in range(ACTION_STEPS):
                    state, reward, done, truncated, info = env.step(action)
                    reward = get_reward(info, reward)
                    state = preprocess_smaller_state(state, device=device)
                    total_reward += reward
                    state = state.unsqueeze(0).unsqueeze(0)
                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    if done:
                        while states.size < N_STEPS:
                            states.append(state)
                            actions.append(action)
                            rewards.append(reward)
                        break

                estimated_reward += value.item() / N_STEPS # the critic estimates the advantage. which is for n steps

                if done or truncated:
                    break

            
            compute_advantages_and_update(global_model,
                                            local_model,
                                            optimizer,
                                            states,
                                            actions,
                                            rewards,
                                            done,
                                            global_model_lock)

            if ONE_LIFE and info['life'] <= 1:
                break


                

        elapsed_time = time.time() - start_time

        with episode_count_lock:
            global_counter.value += 1
            current_episode = global_counter.value
            record_info_for_worker(LOG_FILE_NAME, current_episode, worker_id, elapsed_time, total_reward, info)
            print(f"Worker {worker_id} finished episode: {current_episode}. Time: {elapsed_time:.2f}. Total reward: {total_reward}. Estimated reward: {estimated_reward}")

        if current_episode % MODEL_SAVE_EPISODES == 0:
            with global_model_lock:
                save_model(global_model, current_episode)
    
    print(f"Worker {worker_id} stopped.")

if __name__ == "__main__":
    print("Main thread started")

    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    input_shape = preprocess_smaller_state(env.reset(), device).shape
    n_actions = env.action_space.n

    global_model = ActorCritic(input_shape, n_actions, True, 2).to(device)
    if LOAD_MODEL_EPISODE != -1:
        load_model(global_model, LOAD_MODEL_EPISODE, SAVE_DIR)
    global_model.share_memory()
    global_model.train()

    optimizer = SharedAdam(global_model.parameters(), lr=LEARNING_RATE)

    # num_workers = 8 #mp.cpu_count() - 8
    # processes = []
    # for worker_id in range(num_workers):
    #     p = mp.Process(target=worker,
    #                 args=(global_model,
    #                       optimizer,
    #                         worker_id,
    #                         'SuperMarioBros-v0',
    #                         n_actions,
    #                         global_counter,
    #                         episode_count_lock,
    #                         global_model_lock,
    #                         stop_flag))
    #     p.start()
    #     processes.append(p)


    
    worker(global_model,
            optimizer,
            1,
            'SuperMarioBros-v0',
            n_actions,
            global_counter,
            episode_count_lock,
            global_model_lock,
            stop_flag)

    def signal_handler(sig, frame):
        print("\nReceived termination signal. Stopping workers...")
        stop_flag.value = True
        for p in processes:
            p.join()
        sys.exit(0)

    def on_press(key):
        if key == keyboard.Key.space:
            print("Spacebar pressed. Stopping workers...")
            stop_flag.value = True
            for p in processes:
                p.join()
            return False


    signal.signal(signal.SIGINT, signal_handler) #catch control+C just in case
    print("Press SPACE to stop workers...")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Keep the main process alive
    while not stop_flag.value:
        #sleep(1)
        pass
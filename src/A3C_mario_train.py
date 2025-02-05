import torch
import torch.optim as optim
import torch.multiprocessing as mp
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from actor_critic.actor_critic import ActorCritic, device
from utils import preprocess_smaller_state
import os
from pathlib import Path

# Define the save directory
SAVE_DIR = Path("saved_models")
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
        gamma=0.9,
        entropy_coef=0.01,
        done = False):


    states = torch.stack(states).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    values = torch.stack(values).squeeze().to(device)

    R = 0

    if not done:
        _, next_value = global_model(states[-1].unsqueeze(0))
        R = next_value.item()
    returns = []

    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32)

    advantages = returns - values

    action_probs, _ = global_model(states)
    log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))
    policy_loss = -(log_probs * advantages.detach()).mean()

    value_loss = 0.5 * (returns - values).pow(2).mean()

    entropy = -(action_probs * torch.log(action_probs)).sum(dim=1).mean()
    entropy_loss = -entropy_coef * entropy

    total_loss = policy_loss + value_loss + entropy_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

def worker(global_model,
           optimizer,
           worker_id,
           env_name,
           n_actions,
           n_steps=20,
           gamma=0.99,
           entropy_coef=0.01,
           save_interval=100):  # Save every 100 global episodes
    env = gym_super_mario_bros.make(env_name)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    local_model = ActorCritic(input_shape, n_actions)
    local_model.load_state_dict(global_model.state_dict())

    while True:  # Keep the worker running indefinitely
        state = preprocess_smaller_state(env.reset())
        done = False

        while not done:
            states, actions, rewards, values = [], [], [], []
            for _ in range(n_steps):
                action_probs, value = local_model(state.unsqueeze(0))
                action = torch.multinomial(action_probs, 1).item()

                next_state, reward, done, truncated, _ = env.step(action)
                next_state = preprocess_smaller_state(next_state)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                values.append(value)

                state = next_state
                if done or truncated:
                    break

            compute_advantages_and_update(global_model,
                                          optimizer,
                                          states,
                                          actions,
                                          rewards,
                                          values,
                                          gamma,
                                          entropy_coef,
                                          done)

            local_model.load_state_dict(global_model.state_dict())

        with lock:
            global_counter.value += 1
            current_episode = global_counter.value

        if current_episode % save_interval == 0:
            with lock:
                save_model(global_model, current_episode)


global_counter = mp.Value('i', 0)  # 'i' for integer
lock = mp.Lock()

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
input_shape = preprocess_smaller_state(env.reset()).shape
n_actions = env.action_space.n

global_model = ActorCritic(input_shape, n_actions)
global_model.share_memory()

optimizer = optim.Adam(global_model.parameters(), lr=1e-4)

num_workers = mp.cpu_count()
processes = []
for worker_id in range(num_workers):
    p = mp.Process(target=worker,
                   args=(global_model, optimizer, worker_id, 'SuperMarioBros-v0', n_actions))
    p.start()
    processes.append(p)

for p in processes:
    p.join()
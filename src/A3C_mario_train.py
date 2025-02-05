import torch
import torch.optim as optim
import torch.multiprocessing as mp
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from actor_critic.actor_critic import ActorCritic, device
from utils import preprocess_smaller_state

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

    states = torch.stack(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    values = torch.stack(values).squeeze()

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
           entropy_coef=0.01):
    env = gym_super_mario_bros.make(env_name)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    local_model = ActorCritic(input_shape, n_actions)
    local_model.load_state_dict(global_model.state_dict())

    state = preprocess_smaller_state(env.reset())
    done = False

    while not done:
        states, actions, rewards, values = [], [], [], []
        for _ in range(n_steps):
            action_probs, value = local_model(state.unsqueeze(0))
            print(action_probs)
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
                                      entropy_coef)

        local_model.load_state_dict(global_model.state_dict())



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


# I have a few questions about the code:
# 1. when one episode of the env ends the entire worker ends. Why? Wouldn't we want it to continue in another episode?
# 2.  about these lines of code: 
# for r in reversed(rewards):
#         R = r + gamma * R
#         returns.insert(0, R)

# why reverse and always insert on the 0 instead of going normally and just pushing in the array?

# 3. states, actions, rewards, values = [], [], [], []
# we always start collecting new experiences until done. Why discard all the previous ones? Why not just push a new one and delete an old one?
# 4. Can we store them on the device (which might be cuda might be cpu)
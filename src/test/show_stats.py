import csv
import matplotlib.pyplot as plt

file_path = 'C:/projects/University/AI_4_course/project/super_mario_bros_AI/src/dqn_simple_movement_one_life_action_steps_models/episodes_log.log'

episodes = []
rewards = []
times = []

with open(file_path, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        episodes.append(int(row['Episode']))
        rewards.append(float(row['Reward']))
        times.append(float(row['Time']))

def group_data(data, group_size=300):
    grouped_means = [0]  # Start with (0,0)
    grouped_maxes = [0]
    grouped_x = [0]

    for i in range(0, len(data), group_size):
        group = data[i:i + group_size]
        if len(group) < group_size // 2:  # Ignore incomplete groups with <50% data
            break
        grouped_means.append(sum(group) / len(group))
        grouped_maxes.append(max(group))
        grouped_x.append(i + group_size)  # X-axis point is at the END of the batch

    return grouped_x, grouped_means, grouped_maxes

grouped_episodes, mean_rewards, max_rewards = group_data(rewards, 1000)
_, mean_lifespans, _ = group_data(times, 1000)

plt.figure(figsize=(10, 5))
plt.plot(grouped_episodes, mean_rewards, label='Mean Reward', color='blue')
plt.plot(grouped_episodes, max_rewards, label='Max Reward', color='red', linestyle='--')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Mean and Max Reward per 1000 Episodes')
plt.ylim(bottom=0)
plt.legend()
plt.grid(True)
plt.savefig('mean_rewards.png')
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(grouped_episodes, mean_lifespans, label='Mean Lifespan', color='orange')
plt.xlabel('Episode')
plt.ylabel('Mean Lifespan (Time)')
plt.title('Mean Lifespan per 1000 Episodes')
plt.ylim(bottom=0)
plt.legend()
plt.grid(True)
plt.savefig('mean_lifetime.png')
plt.close()

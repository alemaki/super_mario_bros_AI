import csv
import matplotlib.pyplot as plt

# Step 1: Read the file
file_path = '/home/alemak/projects/AI_4_course/project/marioAI/src/dqn_simple_movement_one_life_smaller_models/episodes_log.log'

# Lists to store data
episodes = []
rewards = []
times = []

with open(file_path, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        episodes.append(int(row['Episode']))
        rewards.append(float(row['Reward']))
        times.append(float(row['Time']))

# Step 2: Group data into bins of 1000 episodes and calculate mean and max rewards
def group_data(data, episodes, group_size=1000):
    grouped_data_mean = []
    grouped_data_max = []
    for i in range(0, len(episodes), group_size):
        group = data[i:i + group_size]
        if group:  # Avoid empty groups
            grouped_data_mean.append(sum(group) / len(group))  # Mean
            grouped_data_max.append(max(group))  # Max
    return grouped_data_mean, grouped_data_max

grouped_episodes = list(range(0, len(episodes), 1000))
mean_rewards, max_rewards = group_data(rewards, episodes, 1000)
mean_lifespans, _ = group_data(times, episodes, 1000)  # We don't need max for lifespans

# Step 3: Plot the mean and max rewards over time
plt.figure(figsize=(10, 5))
plt.plot(grouped_episodes, mean_rewards, label='Mean Reward', color='blue')
plt.plot(grouped_episodes, max_rewards, label='Max Reward', color='red', linestyle='--')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Mean and Max Reward per 1000 Episodes')
plt.ylim(bottom=0)  # Ensure y-axis starts at 0
plt.legend()
plt.grid(True)
plt.savefig('mean_rewards.png')  # Save the plot to a file
plt.close()  # Close the figure to free memory

# Step 4: Plot the average lifespan of Mario over time
plt.figure(figsize=(10, 5))
plt.plot(grouped_episodes, mean_lifespans, label='Mean Lifespan', color='orange')
plt.xlabel('Episode')
plt.ylabel('Mean Lifespan (Time)')
plt.title('Mean Lifespan per 1000 Episodes')
plt.ylim(bottom=0)  # Ensure y-axis starts at 0
plt.legend()
plt.grid(True)
plt.savefig('mean_lifetime.png')  # Save the plot to a file
plt.close()  # Close the figure to free memory


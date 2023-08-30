import matplotlib.pyplot as plt
import numpy as np

# Replace these with your actual reward lists for 5 trials
trial_rewards = [
    [1, 2, 3, 4, 5],   # Trial 1 rewards
    [2, 3, 4, 5, 6],   # Trial 2 rewards
    [3, 4, 5, 6, 7],   # Trial 3 rewards
    [4, 5, 6, 7, 8],   # Trial 4 rewards
    [5, 6, 7, 8, 9]    # Trial 5 rewards
]

# Calculate average rewards and variance over time steps
average_rewards = np.mean(trial_rewards, axis=0)
variance_rewards = np.var(trial_rewards, axis=0)

# Time steps (assuming same length for all trials)
time_steps = np.arange(len(average_rewards))

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the variance as a blurred color background
plt.fill_between(time_steps, average_rewards - variance_rewards, average_rewards + variance_rewards, color='skyblue', alpha=0.3)

# Plot the average reward line
plt.plot(time_steps, average_rewards, marker='o', color='blue', label='Average Reward')

plt.title('Average Reward and Variance over Time')
plt.xlabel('Time Step')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)

# Save the plot as a PNG image
plt.savefig('average_reward_variance_plot.png')

# Display the plot (optional)
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = make_vec_env("CartPole-v1", n_envs=1)

# Create and train the A2C agent
model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

# Collect the episode rewards during training
episode_rewards = []
episodes = 1000  # Number of episodes to run for data collection

for _ in range(episodes):
    obs = env.reset()
    total_reward = 0

    while True:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            episode_rewards.append(total_reward)
            break

# Plot the training rewards
episode_numbers = np.arange(1, len(episode_rewards) + 1)

plt.plot(episode_numbers, episode_rewards)
plt.xlabel('Episode Number')
plt.ylabel('Episode Reward')
plt.title('Training Reward over Episodes')

# Save the plot as an image file (e.g., PNG)
plt.savefig('/home/s223540177/dai/SimpleRL/A2C_/figs/sb3_training_rewards_plot.png')

# Show the plot (optional)
plt.show()

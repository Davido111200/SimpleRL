import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = make_vec_env("CartPole-v1", n_envs=1)

# Create and train the A2C agent
model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

def plot_reward_trials_with_variance(trial_scores, filename, blurred_variance_factor=0.3):
    """
    Plots multiple reward trials with an average line and blurred variance.

    Args:
        trial_scores (list of arrays): List containing arrays of trial scores for each trial.
        blurred_variance_factor (float): Factor controlling the amount of blurring for variance.

    Returns:
        None
    """
    num_trials = len(trial_scores)

    # Calculate the average and variance
    average_scores = np.mean(trial_scores, axis=0)
    blurred_variance = np.std(trial_scores, axis=0) * blurred_variance_factor

    # Plot the average line
    plt.plot(average_scores, label='Average', color='blue')

    # Plot the blurred variance area
    plt.fill_between(range(len(average_scores)), average_scores - blurred_variance, average_scores + blurred_variance, alpha=0.3, color='blue')

    # Plot individual trial scores with different colors
    
    colors = ['red', 'green', 'orange', 'purple', 'brown']
    for i, color in zip(range(num_trials), colors):
        plt.plot(trial_scores[i], color=color, alpha=0.3)

    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.title('Reward Trials with Average and Blurred Variance')
    plt.savefig(filename)
    plt.legend()
    plt.show()

# Collect the episode rewards during training
trial_scores = []
episodes = 1000  # Number of episodes to run for data collection
n_trials = 5
max_ts = 1000
for trial in range(n_trials):
    scores = []
    for _ in range(episodes):
        obs = env.reset()
        total_reward = 0

        for ts in range(max_ts):
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward

            if done:
                break
        scores.append(ts)

    trial_scores.append(scores)

# Plot the training rewards

plot_reward_trials_with_variance(trial_scores, '/home/s223540177/dai/SimpleRL/A2C_/figs/a2c_training_rewards.png')
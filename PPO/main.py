import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import namedtuple
from ppo import PPO

import wandb  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

Transition = namedtuple('transition', ('state', 'action', 'reward', 'next_state'))

def plot_multiple_trials_with_average_and_std(all_trials_rewards):
    all_trials_rewards = np.array(all_trials_rewards)
    mean = np.mean(all_trials_rewards, axis=0)
    std = np.std(all_trials_rewards, axis=0)
    plt.plot(mean, label="mean")
    plt.fill_between(np.arange(len(mean)), mean-std, mean+std, alpha=0.3)
    plt.savefig("../PPO/figs/Pendulum_v1/ppo.png")
    plt.show()

def main(n_epochs, max_timesteps, n_trials, env_name, epsilon, batch_size):
    env = gym.make(env_name)
    print(device)
    wandb.init(project="PPO")
    all_trials_rewards = []
    for trial in range(n_trials):
        model = PPO(env=env, batch_size=batch_size, epsilon=epsilon)
        rewards = model.learn(n_epochs, batch_size)
        all_trials_rewards.append(rewards)
    plot_multiple_trials_with_average_and_std(all_trials_rewards)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--max_timesteps", type=int, default=1000)
    parser.add_argument("--n_trials", type=int, default=3)
    parser.add_argument("--env_name", type=str, default="Pendulum-v1")
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    main(args.n_epochs, args.max_timesteps, args.n_trials, args.env_name, args.epsilon, args.batch_size)
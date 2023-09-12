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


from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env

import wandb  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

Transition = namedtuple('transition', ('state', 'action', 'reward', 'next_state'))

def plot_reward_trials_with_variance(trial_scores, filename, blurred_variance_factor=0.3):
    """
    Plots multiple reward trials with an average line and blurred variance.

    Args:
        trial_scores (list of arrays): List containing arrays of trial scores for each trial.
        blurred_variance_factor (float): Factor controlling the amount of blurring for variance.

    Returns:
        None
    """
    plt.clf()
    for trial in trial_scores:
        plt.plot(trial, alpha=0.3)
    blurred_variance = np.mean(trial_scores, axis=0)
    blurred_variance = np.convolve(blurred_variance, np.ones(10)/10, mode='same')
    plt.plot(blurred_variance, label="mean")
    plt.fill_between(np.arange(len(blurred_variance)), blurred_variance-blurred_variance_factor, blurred_variance+blurred_variance_factor, alpha=0.3)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward Trials')
    plt.savefig(filename)
    plt.show()


def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


def main(n_epochs, n_cpu, n_trials, env_name, epsilon, batch_size):
    # should have parallel environments
    # create the vectorized environment
    env = SubprocVecEnv([make_env(env_name, i) for i in range(n_cpu)])

    all_trials_rewards = []
    for trial in range(n_trials):
        model = PPO(env=env, batch_size=batch_size, epsilon=epsilon)
        rewards = model.learn(n_epochs, batch_size)
        all_trials_rewards.append(rewards)
    plot_reward_trials_with_variance(all_trials_rewards, "../PPO/figs/Hopper/ppo.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=600)
    parser.add_argument("--n_cpu", type=int, default=4)
    parser.add_argument("--n_trials", type=int, default=3)
    parser.add_argument("--env_name", type=str, default="Hopper-v4")
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args.n_epochs, args.n_cpu, args.n_trials, args.env_name, args.epsilon, args.batch_size)
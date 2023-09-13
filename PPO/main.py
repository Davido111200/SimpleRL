import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import namedtuple
from ppo_refactor import PPO

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env

import wandb  

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# now can support cpu only
device = torch.device("cpu")

torch.autograd.set_detect_anomaly(True)

def main(n_epochs, env_name, n_envs, n_step_per_batch, epsilon, vf_coef, ent_coef):
    # init wandb
    wandb.init(project="ppo")
    # should have parallel environments
    # create the vectorized environment
    batch_size = n_envs * n_step_per_batch

    model = PPO(n_epochs, env_name, n_envs, n_step_per_batch, batch_size, epsilon, vf_coef, ent_coef)
    model.learn()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=3000000)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--env_name", type=str, default="Walker2d-v4")
    parser.add_argument("--n_step_per_batch", type=int, default=128)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    args = parser.parse_args()
    main(args.n_epochs, args.env_name, args.n_envs, args.n_step_per_batch, args.epsilon, args.vf_coef, args.ent_coef)
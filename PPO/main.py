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

class Buffer(object):
    def __init__(self, capacity) -> None:
        self.memory = []
        self.capacity = capacity
    
    def add(self, *args):
        """
        Save a transition to buffer
        """
        self.memory.append(Transition(*args))

    def buffer_size_check(self):
        return True if len(self.memory)==self.capacity else False

    def reset(self):
        self.memory = []
        return self.memory

def main(n_epochs, max_timesteps, n_trials, env_name, epsilon, batch_size):
    env = gym.make(env_name)
    # wandb.init(project="PPO")
    model = PPO(env=env)
    model.rollout(batch_size)





    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--max_timesteps", type=int, default=1000)
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--env_name", type=str, default="Hopper-v4")
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()
    main(args.n_epochs, args.max_timesteps, args.n_trials, args.env_name, args.epsilon, args.batch_size)
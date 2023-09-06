import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import argparse

from torch.distributions import Categorical
import wandb  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

class Policy(nn.Module):
    "Simple policy pi"
    def __init__(self, n_inputs, n_outputs) -> None:
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 128, device=device, dtype=torch.float32)
        self.fc2 = nn.Linear(128, n_outputs, device=device, dtype=torch.float32)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)

class Value_function(nn.Module):
    "Simple value function"
    def __init__(self, n_inputs) -> None:
        super(Value_function, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 128, device=device, dtype=torch.float32)
        self.fc2 = nn.Linear(128, 1, device=device, dtype=torch.float32)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
    
def main(n_epochs, max_timesteps, n_trials, env_name, epsilon):
    env = gym.make(env_name)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--max_timesteps", type=int, default=1000)
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--env_name", type=str, default="Hopper-v1")
    parser.add_argument("--epsilon", type=float, default=0.2)
    args = parser.parse_args()
    main(args.n_epochs, args.max_timesteps, args.n_trials, args.env_name, args.epsilon)
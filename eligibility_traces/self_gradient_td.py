from env import grid_world
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cpu") # sad

class policy_network(nn.Module):
    """
    This module might change according to the specific neural networks requirements of
    different algorithms, so the design here is minimal
    """
    def __init__(self, n_inputs, n_outputs) -> None:
        super(policy_network, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.fc1 = nn.Linear(self.n_inputs, 512, device=device)
        self.fc2 = nn.Linear(512, self.n_outputs, device=device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x) # stands for action probabilities
    
def main(n_eps, max_ts):
    # HYPERS
    GAMMA = 0.9
    ALPHA = 0.9
    env = grid_world(n_rows=10, n_cols=10)

    # init 
    policy_net = policy_network(env.n_observations, env.n_actions)
    optimizer = optim.AdamW(policy_net.parameters())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("max_ts", type=int, default=100)

    args = parser.parse_args()

    main(args.n_epochs, args.max_ts)

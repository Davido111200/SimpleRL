import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from short_corridor import short_env
import argparse

device = torch.device("cpu")

class policy_network(nn.Module):
    def __init__(self, n_inputs, hidden, n_outputs) -> None:
        super(policy_network, self).__init__()
        self.n_inputs = n_inputs
        self.hidden = hidden
        self.n_outputs = n_outputs
        self.fc1 = nn.Linear(self.n_inputs, self.hidden, device=device)
        self.fc2 = nn.Linear(self.hidden, self.n_outputs, device=device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def main(n_epochs):
    env = short_env(n_states=4)

    policy = policy_network(n_inputs=env.n_states, hidden=128, n_outputs=env.n_actions)

    ALPHA = 0.9
    GAMMA = 0.9

    for epoch in range(n_epochs):
        trajectory = []
        state, terminated = env.reset()
        while not terminated:
            # generate a whole trajectory
            pass
            

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, device=device)

    arguments = parser.parse_args()

    main(arguments.n_epochs)
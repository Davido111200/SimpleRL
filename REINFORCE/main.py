import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from ..grid_world import grid_world

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

def main():
    env = grid_world(n_rows=)
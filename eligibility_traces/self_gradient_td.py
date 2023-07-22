from env import grid_world
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class neural_net(nn.Module):
    """
    This module might change according to the specific neural networks requirements of
    different algorithms, so the design here is minimal
    """
    def __init__(self, n_inputs, n_outputs) -> None:
        super(neural_net, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.fc1 = nn.Linear(self.n_inputs, 512, device=device)
        self.fc2 = nn.Linear(512, self.n_outputs, device=device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
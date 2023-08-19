import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import random
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SARSA(nn.Module):
    def __init__(self, n_actions, hor, ver) -> None:
        super(SARSA, self).__init__()
        self.q_tables = torch.rand(size=(hor*ver, n_actions))

    def update(self, pos, target_val, lr, act):
        self.q_tables[pos, act] = self.q_tables[pos, act] + lr * (target_val - self.q_tables[pos, act])

    def get(self):
        return self.q_tables 
    


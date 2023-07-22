import gymnasium as gym
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cpu") # sad :(

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

class grid_world(gym.Env):
    def __init__(self, n_rows, n_cols) -> None:
        super(grid_world, self).__init__()
        self.n_rows = n_rows
        self.n_cols = n_cols 
        self.grid = torch.zeros(shape=(self.n_rows, self.n_cols))
        self.starting_pos = torch.tensor([0, 0]) # fixed starting point
        self.target_pos = torch.tensor([1, 1]) # fixed target - change later
        self.terminated = False
        self.cur_pos = torch.tensor([0, 0]) # current position - fixed as the starting point

    def reset(self):
        self.cur_pos, self.terminated = torch.tensor([0, 0]), False
        return self.cur_pos, self.terminated
    
    def check_pos(self):
        """
        check out of bound positions
        """
        cur_row, cur_col = self.cur_pos[0], self.cur_pos[1]

        if cur_row < 0:
            new_row = 0
        elif cur_row >= self.n_rows:
            new_row = self.n_rows
    
        if cur_col < 0:
            new_col = 0
        elif cur_col >= self.n_cols:
            new_col = self.n_cols
        
        # return a flag to see if agent violates the environment
        if cur_row == new_row and cur_col == new_col:
            flag = True
        else:
            flag = False
        
        return torch.tensor([new_row, new_col]), flag
        
    def make_a_move(self, action):
        """
        make a move based on the action selected
        WARNING: new postion of this methods might contain out of bound elements -> needs checking after
        """
        if action == 0:
            new_pos = torch.tensor([self.cur_pos[0], self.cur_pos[1] - 1])
        elif action == 1:
            new_pos = torch.tensor([self.cur_pos[0], self.cur_pos[1] + 1])
        elif action == 2:
            new_pos = torch.tensor([self.cur_pos[0] - 1, self.cur_pos[1]])
        elif action == 3:
            new_pos = torch.tensor([self.cur_pos[0] + 1, self.cur_pos[1]])

        return new_pos

    def action_selection(self, threshold):
        """
        current policy follow epsilon-greedy
        """
        temp = random.random() 
        if temp < threshold:
            # return a randomly selected action ( current sample is from 4 actions)
            return torch.randint(low=0, high=4, size=(1, ))
        else:
            # based on the function approximation, choose the greedy action
            pass        


    def step(self, action):
        """
        action: there are total of 4 actions: left, right, up, down
        correspond to [0, 1, 2, 3]
        """


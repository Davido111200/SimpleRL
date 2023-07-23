import gymnasium as gym
import numpy as np
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cpu") # sad :(

class grid_world(gym.Env):
    def __init__(self, n_rows, n_cols) -> None:
        super(grid_world, self).__init__()
        self.n_rows = n_rows
        self.n_cols = n_cols 
        self.grid = torch.zeros(size=(self.n_rows, self.n_cols))
        self.starting_pos = torch.tensor([0, 0]) # fixed starting point
        self.target_pos = torch.tensor([1, 1]) # fixed target - change later
        self.terminated = False
        self.cur_pos = torch.tensor([0, 0]) # current position - fixed as the starting point
        self.n_observations = 2
        self.n_actions = 4

    def reset(self):
        self.cur_pos, self.terminated = torch.tensor([0., 0.]), False
        return self.cur_pos, self.terminated
    
    def sample(self):
        """
        randomly select an action - can be seen as the behavior policy
        """
        return random.randint(0, 3)
    
    def check_pos(self):
        """
        check out of bound positions
        """
        cur_row, cur_col = self.cur_pos[0], self.cur_pos[1]

        if cur_row < 0:
            new_row = 0
        elif cur_row >= self.n_rows:
            new_row = self.n_rows
        else:
            new_row = cur_row
    
        if cur_col < 0:
            new_col = 0
        elif cur_col >= self.n_cols:
            new_col = self.n_cols
        else:
            new_col = cur_col
        
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

        # update the new_position
        self.cur_pos = new_pos

    def action_selection(self, threshold, policy, state):
        """
        current policy follow epsilon-greedy
        """
        temp = random.random() 
        if temp < threshold:
            # return a randomly selected action ( current sample is from 4 actions)
            return torch.tensor(self.sample(), dtype=torch.int64, device=device)
        else:
            # based on the function approximation, choose the greedy action
            return torch.argmax(policy(state))


    def step(self, action):
        """
        action: there are total of 4 actions: left, right, up, down
        correspond to [0, 1, 2, 3]
        """
        action = action.item()

        self.make_a_move(action)
        next_state, flag = self.check_pos()

        terminated = torch.all(next_state.eq(self.target_pos))
        reward = 1 if terminated else 0

        return next_state, reward, terminated, False, {}
    





import numpy as np

import torch
import torch.nn as nn

device = torch.device("cpu")


class short_corridor(object):
    def __init__(self, n_states) -> None:
        self.n_states = n_states
        self.grid = np.zeros(shape=(self.n_states, 1), device=device)
        self.start_pos = 0 # starting state is always 0
        self.current_pos = 0
        self.flag = False # at the starting state, flag is false
        self.terminated = False

    def check_current_state(self, current_pos):
        if current_pos < 0:
            current_pos = 0
            terminated = False
        elif current_pos >= self.n_states:
            current_pos = self.n_states - 1
            terminated = True
        else:
            current_pos = self.n_states
            terminated = False

        # update the current pos to self.current_pos
        self.current_pos = current_pos
        return current_pos, terminated

    def make_a_move(self, action, flag):
        "flag is raised when we are in special state"
        if action == 0:
            if flag:
                temp = self.current_pos + 1
                self.current_pos, self.terminated = self.check_current_state(temp)
            else:
                temp = self.current_pos - 1
                self.current_pos, self.terminated = self.check_current_state(temp)
        elif action == 1:
            if flag:
                temp = self.current_pos - 1
                self.current_pos, self.terminated = self.check_current_state(temp)
            else:
                temp = self.current_pos + 1
                self.current_pos, self.terminated = self.check_current_state(temp)


        return self.current_pos, self.terminated

    def step(self, action):
        """
        action only in [0, 1], where 0 is a left and 1 is a right action
        """
        
        # check for special state
        if self.current_pos == 1:
            self.flag = True

        next_state, terminated = self.make_a_move(action, self.flag)
        reward = 1 if terminated else -1

        self.current_pos = next_state

        return next_state, reward, terminated
    
    def reset(self):
        self.current_pos, self.terminated, self.flag = self.start_pos, False, False
        return self.current_pos, self.terminated
    






        
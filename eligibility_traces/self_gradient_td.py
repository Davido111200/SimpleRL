from env import grid_world
import numpy as np
import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cpu") # sad

class function_approximation(nn.Module):
    """
    This module might change according to the specific neural networks requirements of
    different algorithms, so the design here is minimal
    Actually, the problem we are dealing with assume the function approximation to be 
    a linear one -> 1 layer only might solve the problem
    If there are more than one layer, the trace should have the dimension of
    the sum of all weights
    """
    def __init__(self, n_inputs, n_hiddens, n_outputs) -> None:
        super(function_approximation, self).__init__()
        self.n_inputs = n_inputs
        self.n_hiddens = n_hiddens
        self.n_outputs = n_outputs
        self.fc1 = nn.Linear(self.n_inputs, self.outputs, device=device)
        # self.fc2 = nn.Linear(self.n_hiddens, self.n_outputs, device=device)

    def forward(self, x):
        # x = F.relu(self.fc1(x))
        x = self.fc1(x)
        return x # stands for action probabilities
    

def main(n_eps, max_ts):
    # HYPERS
    GAMMA = 0.9
    ALPHA = 0.9
    LAMBDA = random.random()
    EPSILON = 0.3
    env = grid_world(n_rows=10, n_cols=10)

    n_inputs = env.n_observations
    n_outputs = env.n_actions
    n_hiddens = 512

    # init 
    value_approximator = function_approximation(env.n_observations, n_hiddens, env.n_actions)
    optimizer = optim.AdamW(value_approximator.parameters())

    # there is only 1 layer of weights
    n_inputs = env.n_observations
    n_outputs = env.n_actions
    n_hidden = 512
    
    # initialize the value-function weights
    w = torch.randn(n_outputs, n_inputs)

    for epoch in range(n_eps):
        state, terminated = env.reset()
        z = torch.zeros((n_outputs, 1))
        while not terminated:
            action = env.action_selection(threshold=EPSILON, policy=policy_net, state=state)
            next_state, reward, terminated, truncated, _ = env.step(action=action.item())

            # forward pass to get the output
            state_values = value_approximator(state)
            state_values.backward()

            # access the weights of the linear layer
            weights = value_approximator.fc1.weight
            gradients_fc1 = value_approximator.fc1.weight.grad

            assert len(gradients_fc1) == len(z), "z shape error"

            # update eligibility trace
            z = GAMMA * LAMBDA * z + gradients_fc1

            # calculate TD(0) error
            td_error = reward + GAMMA * value_approximator(next_state, weights) - value_approximator(state, weights)

            # update weights
            weights += ALPHA * td_error * z

            # iterate
            state = next_state
            







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("max_ts", type=int, default=100)

    args = parser.parse_args()

    main(args.n_epochs, args.max_ts)

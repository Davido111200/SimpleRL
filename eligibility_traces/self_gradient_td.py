from env import grid_world
import numpy as np
import argparse
import random
import matplotlib.pyplot as plt
import matplotlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cpu") # sad
plotter = []

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
        self.fc1 = nn.Linear(self.n_inputs, self.n_outputs, device=device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return x # stands for action probabilities
    

def main(n_eps, max_ts):
    # HYPERS
    GAMMA = 0.9
    ALPHA = 0.9
    LAMBDA = random.random()
    EPSILON = 0.3
    env = grid_world(n_rows=10, n_cols=10)

    n_inputs = env.n_observations
    n_outputs = 1
    n_hiddens = 512

    # init 
    value_approximator = function_approximation(env.n_observations, n_hiddens, n_outputs)
    optimizer = optim.AdamW(value_approximator.parameters())

    # initialize the value-function weights
    w = torch.randn(n_outputs, n_inputs)

    for epoch in range(n_eps):
        state, terminated = env.reset()
        z = torch.zeros((n_outputs, 1))
        env.total_ts = 0
        while not terminated:
            # check for maximum timestep condition
            env.total_ts += 1
            if env.total_ts > max_ts:
                print(env.total_ts)
                print("Maximal timesteps reached")
                break

            action = env.action_selection(threshold=EPSILON, policy=value_approximator, state=state)
            next_state, reward, terminated, truncated, _ = env.step(action=action)

            # forward pass to get the output
            state_value = value_approximator(state)
            
            # now we want to obtain the gradients of v w.r.t. w.
            # however, as mentioned in the setting, the function approximation we are using is linear
            # thus, the gradient in terms of func approximation is X, which is exactly as the input to neural net
            # updating eligibility trace
            z =  GAMMA * LAMBDA * z + state
            
            weights = value_approximator.fc1.weight
            
            next_state_value = value_approximator(next_state)

            # calculate TD(0) error
            td_error = reward + GAMMA * next_state_value - state_value

            # update weights
            weights = weights + ALPHA * td_error * z

            # iterate
            state = next_state

            plotter.append(td_error.detach().numpy())

    print("Finished training")

def plotting(y):
    x = range(len(y))

    plt.plot(x, y, marker='o', linestyle='-', color='b')

    plt.xlabel('Index')
    plt.ylabel('Data Points')
    plt.title('Data Points Plot')

    # Show the plot
    plt.show()
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--max_ts", type=int, default=1000)

    args = parser.parse_args()

    main(args.n_epochs, args.max_ts)
    plotting(plotter)

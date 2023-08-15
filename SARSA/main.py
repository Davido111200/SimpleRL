import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import random
import math
import gymnasium as gym
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class neural_net(nn.Module):
    def __init__(self, n_inputs, n_outputs) -> None:
        super(neural_net, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 512, device=device, dtype=torch.float64)
        self.fc2 = nn.Linear(512, n_outputs, device=device, dtype=torch.float64)

    def encode(self, state, state_size):
        temp = torch.zeros(size=state_size, dtype=torch.float64, device=device)
        temp[state] = 1.0
        return temp

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.argmax(F.softmax(self.fc2(x))), F.softmax(self.fc2(x))
    

def main(n_epochs, max_ts):
    env = gym.make('CartPole-v1')
    net = neural_net(env.observation_space.shape[0], env.action_space.n, device=device)

    def action_selection(eps, state):
        temp = random.random()
        if temp < eps:
            return torch.randint(low=0, high=env.action_space.n - 1)
        else:
            state_tensor = net.encode(state)
            return net(state_tensor)[0]
        
    def anneal_eps():
        pass


    for epoch in range(n_epochs):
        state, _ = env.reset()
        terminated = False

        for ts in range(max_ts):
            action = action_selection()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--max_ts", type=int, default=1000)

    args = parser.parse_args()
    main(args.n_epochs, args.max_ts)
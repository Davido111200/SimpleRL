import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import random
import gymnasium as gym
import argparse

env = gym.make("CartPole-v1")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class policy(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(policy, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 128, device=device, dtype=torch.float32)
        self.fc2 = nn.Linear(128, n_outputs, device=device, dtype=torch.float32)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(x, dim=0)

def main(n_epochs, max_ts, n_trials):
    pi = policy(env.observation_space.shape[0], env.action_space.n)
    optimizer = optim.Adam(pi.parameters(), lr=0.01)
    step_size = 0.01
    gamma = 0.99

    states = []
    actions = []
    rewards = []
    log_probs = []

    for epoch in range(n_epochs):
        state, _ = env.reset()
        state = torch.from_numpy(state).float().to(device)
        done = False
        
        # getting the full trajectory
        for ts in range(max_ts):
            action_probs = pi(state)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()

            next_state, reward, done, truncated, _ = env.step(action.item())
            
            log_prob = torch.log(action_probs[action])

            log_probs.append(log_prob)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            if done:
                break
                
            state = next_state
        
        # update right away
        for t in range(len(states)):
            G = 0
            for k in range(t, len(states)):
                G += gamma**(k-t) * rewards[k]
                loss = -log_probs[t] * G * step_size * (gamma**t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # print the results
        if epoch % 10 == 0:
            print("Epoch: {}, Reward: {}".format(epoch, sum(rewards)))
            

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--max_ts", type=int, default=1000)
    parser.add_argument("--n_trials", type=int, default=5)
    args = parser.parse_args()
    main(args.n_epochs, args.max_ts, args.n_trials)
    
    


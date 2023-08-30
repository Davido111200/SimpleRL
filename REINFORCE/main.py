import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import random
import gymnasium as gym
import argparse
import matplotlib.pyplot as plt
from collections import deque


env = gym.make("CartPole-v1")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class policy(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(policy, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 32, device=device, dtype=torch.float32)
        self.fc2 = nn.Linear(32, n_outputs, device=device, dtype=torch.float32)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)
    
def plot_scores(scores, filename):
    fig = plt.figure()
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

def main(n_epochs, max_ts, n_trials):
    pi = policy(env.observation_space.shape[0], env.action_space.n)
    optimizer = optim.Adam(pi.parameters(), lr=0.01)
    gamma = 0.99


    for trial in range(n_trials):
        scores = []
        running_reward = 10
        for epoch in range(n_epochs):
            states = []
            actions = []
            rewards = []
            log_probs = []
            state, _ = env.reset()
            done = False
            
            # Getting the full trajectory
            for ts in range(max_ts):
                state = torch.from_numpy(state).float().to(device)
                action_probs = pi(state)
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()

                next_state, reward, done, truncated, _ = env.step(action.item())
                
                # this was the problem the whole time!!!!!!
                log_prob = action_dist.log_prob(action)

                log_probs.append(log_prob)
                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state
                
                if done:
                    break
                                
            R = 0
            returns = []
            policy_loss = []
            for r in rewards[::-1]:
                R = r + gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns)
            # returns = (returns - returns.mean()) / (returns.std() + 1e-7)

            for lp, Gt in zip(log_probs, returns):
                policy_loss.append(-lp * Gt)
            
            optimizer.zero_grad()
            policy_loss = sum(policy_loss)
            policy_loss.backward()
            optimizer.step()


            running_reward = running_reward * 0.99 + ts * 0.01

            if epoch % 100 == 0:
                print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                    epoch, ts, running_reward))
            if running_reward > env.spec.reward_threshold:
                print("Solved! Running reward is now {} and "
                    "the last episode runs to {} time steps!".format(running_reward, ts))
                break
        
        plot_scores(scores, filename="/home/s223540177/dai/SimpleRL/REINFORCE/figs/trial_{}.png".format(trial))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=10000)
    parser.add_argument("--max_ts", type=int, default=1000)
    parser.add_argument("--n_trials", type=int, default=2)
    args = parser.parse_args()
    main(args.n_epochs, args.max_ts, args.n_trials)
    
    


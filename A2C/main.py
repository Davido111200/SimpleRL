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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    "Simple Actor Network - takes in state and outputs action probabilities"
    def __init__(self, n_inputs, n_outputs):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 128, device=device, dtype=torch.float32)
        self.fc2 = nn.Linear(128, n_outputs, device=device, dtype=torch.float32)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)

class Critic(nn.Module):
    "Simple Critic Network - takes in state and outputs state value"
    def __init__(self, n_inputs):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 128, device=device, dtype=torch.float32)
        self.fc2 = nn.Linear(128, 1, device=device, dtype=torch.float32)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

def main(n_epochs, max_ts, n_trials):
    trial_scores = []
    env = gym.make("CartPole-v1")

    for trial in range(n_trials):
        scores = []
        actor = Actor(env.observation_space.shape[0], env.action_space.n).to(device)
        actor_optimizer = optim.AdamW(actor.parameters(), lr=0.01)
        critic = Critic(env.observation_space.shape[0]).to(device)
        critic_optimizer = optim.AdamW(critic.parameters(), lr=0.01)

        gamma = 0.99

        step_size_actor = 0.9
        step_size_critic = 0.9

        running_reward = 10

        for epoch in range(n_epochs):
            # init first state
            state, _ = env.reset()
            I = 1
            state = torch.from_numpy(state).float().to(device)
            for ts in range(max_ts):
                action_probs = actor(state)
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()

                next_state, reward, terminated, truncated, _ = env.step(action.item())
                
                log_prob_action = action_dist.log_prob(action).to(device)

                current_state_value = critic(state)

                if terminated:
                    td_error = reward - current_state_value.item()
                else:
                    next_state = torch.from_numpy(next_state).float().to(device)
                    next_state_value = critic(next_state)
                    td_error = reward + gamma * next_state_value.item() - current_state_value.item()
                
                # # critic update
                # loss_critic = - td_error * step_size_critic * current_state_value
                # critic_optimizer.zero_grad()
                # loss_critic.backward()
                # critic_optimizer.step()

                # # actor update
                # loss_actor = - td_error * step_size_actor * I * log_prob_action
                # actor_optimizer.zero_grad()
                # loss_actor.backward()
                # actor_optimizer.step()

                I *= gamma
                state = next_state

                if terminated:
                    break

            running_reward = running_reward * 0.99 + ts * 0.01
            scores.append(running_reward)

            if epoch % 100 == 0:
                print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                    epoch, ts, running_reward))
            if running_reward > env.spec.reward_threshold:
                print("Solved! Running reward is now {} and "
                    "the last episode runs to {} time steps!".format(running_reward, ts))
        trial_scores.append(scores)


        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Actor-Critic')
    parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--max_ts', type=int, default=1000, help='max time steps')
    parser.add_argument('--n_trials', type=int, default=2, help='number of trials')
    args = parser.parse_args()
    main(args.n_epochs, args.max_ts, args.n_trials)
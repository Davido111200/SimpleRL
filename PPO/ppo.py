import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as distributions

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import namedtuple

from torch.distributions import Categorical
import wandb  

device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')

class Neural_net(nn.Module):
    "Simple policy and value function"
    def __init__(self, n_inputs, n_outputs):
        super(Neural_net, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 128, device=device, dtype=torch.float32)
        self.fc2 = nn.Linear(128, n_outputs, device=device, dtype=torch.float32)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class PPO:
    def __init__(self, env) -> None:
        self.actor = Neural_net(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
        self.critic = Neural_net(env.observation_space.shape[0], 1).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.env = env 
        self.discount_factor = 0.99

        # initialize the covanriance matrix used to query the actor for actions
        self.cov_var = torch.full(size=(env.action_space.shape[0],), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var).to(device)

    def get_action(self, state):
        """
        Return action, by sample from distribution with:
        mean    : (n_action,)
        variance: predefined above 
        """
        mean = self.actor(state)

        dist = distributions.MultivariateNormal(loc=mean, covariance_matrix=self.cov_mat)

        action = dist.sample()

        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()

    def compute_rtg(self, rewards, discount_factor):
        """
        Return rewards-to-go for each time-step
        We go reverse to get the full trajectory rtgs
        """
        rtgs = []

        discounted_reward = 0

        for rew in reversed(rewards):
            discounted_reward = rew + discount_factor * discounted_reward
            rtgs.append(discounted_reward)
        
        return rtgs


    def rollout(self, batch_size):
        """
        Get the batch data
        """
        batch_state = []
        batch_reward = []
        batch_action = []
        batch_next_state = []
        batch_log_action = []
        batch_rtg = []

        for batch in range(batch_size):
            state, _ = self.env.reset()
            done = False
            
            # we do not directly save epoch reward, since we want the rewards-to-go for each timestep
            eps_reward = []
            while not done:
                action, log_prob = self.get_action(torch.as_tensor(state, dtype=torch.float32, device=device))

                next_state, reward, terminated, truncated, _ = self.env.step(action)

                batch_state.append(state)
                eps_reward.append(reward)
                batch_action.append(action)
                batch_next_state.append(next_state)
                batch_log_action.append(log_prob)

                done = terminated or truncated

                if done:
                    break
            batch_reward.append(eps_reward)
            batch_rtg.append(self.compute_rtg(eps_reward, discount_factor=self.discount_factor))
            
            # convert to tensors
            batch_state = torch.tensor(batch_state, dtype=torch.float32, device=device)
            batch_action = torch.tensor(batch_action, dtype=torch.float32, device=device)
            batch_next_state = torch.tensor(batch_next_state, dtype=torch.float32, device=device)
            batch_log_action = torch.tensor(batch_log_action, dtype=torch.float32, device=device)
            batch_rtg = torch.as_tensor(batch_rtg, dtype=torch.float32, device=device)
            batch_reward = torch.as_tensor(batch_reward, dtype=torch.float32, device=device)

            print(batch_log_action.requires_grad)
            quit()
        
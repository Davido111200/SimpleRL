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
    def __init__(self, env, batch_size, epsilon) -> None:
        self.actor = Neural_net(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
        self.critic = Neural_net(env.observation_space.shape[0], 1).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.env = env 
        self.discount_factor = 0.99
        self.batch_size = batch_size
        self.epsilon = epsilon

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

        return action.detach().numpy(), log_prob.detach(), dist

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
                action, log_prob, dist = self.get_action(torch.as_tensor(state, dtype=torch.float32, device=device))

                next_state, reward, terminated, truncated, _ = self.env.step(action)

                batch_state.append(state)
                eps_reward.append(reward)
                batch_action.append(action)
                batch_next_state.append(next_state)
                batch_log_action.append(log_prob)

                done = terminated or truncated

                state = next_state
                if done:
                    break
            batch_reward.append(eps_reward)
            batch_rtg.append(self.compute_rtg(eps_reward, discount_factor=self.discount_factor))
            
            # first convert to numpy array for batch_state
            batch_state = np.array(batch_state)
            batch_action = np.array(batch_action)
            batch_next_state = np.array(batch_next_state)
            batch_log_action = np.array(batch_log_action)
            batch_rtg = np.array(batch_rtg)
            batch_reward = np.array(batch_reward)


            # convert to tensors
            batch_state = torch.tensor(batch_state, dtype=torch.float32, device=device)
            batch_action = torch.tensor(batch_action, dtype=torch.float32, device=device)
            batch_next_state = torch.tensor(batch_next_state, dtype=torch.float32, device=device)
            batch_log_action = torch.tensor(batch_log_action, dtype=torch.float32, device=device)
            batch_rtg = torch.as_tensor(batch_rtg, dtype=torch.float32, device=device)
            batch_reward = torch.as_tensor(batch_reward, dtype=torch.float32, device=device)

            return batch_state, batch_action, batch_log_action, batch_reward, batch_rtg, batch_next_state, dist
    
    def get_log_prob(self, batch_state, batch_action):
        """
        Compute the log prob of current data batch
        """
        mean = self.actor(batch_state)
        dist = torch.distributions.MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_action)

        return log_probs


    def learn(self, learn_steps, batch_size):
        t_steps = 0
        while t_steps < learn_steps:
            # rollout to take the data
            # the data is batched with number of samples = batch_size
            t_steps += 1
            batch_state, batch_action, batch_log_action, batch_reward, batch_rtg, batch_next_state, dist = self.rollout(batch_size)

            # compute advantage
            A_k = batch_rtg - self.critic(batch_state).squeeze()

            # normalize advantage
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # compute the loss
            # actor loss
            # the new policy probability = taking the probs with current actor for the action considering
            
            # compute the log probability for current policy
            current_log_action = self.get_log_prob(batch_state, batch_action)
            
            # the old log probability is calculated in rollout
            # now we calculate the ratio
            ratio = torch.exp(current_log_action - batch_log_action)

            surrogate_loss = - torch.sum(A_k * torch.min(ratio, torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)))

            self.actor_optimizer.zero_grad()
            surrogate_loss.backward()
            self.actor_optimizer.step()

            # critic loss
            # calculate estimated value
            target_state_value = batch_reward.T + self.critic(batch_next_state) * self.discount_factor
            current_state_value = self.critic(batch_state)
            
            # we use MSE loss
            criterion = nn.MSELoss()
            critic_loss = criterion(current_state_value, target_state_value.detach())

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            wandb.log({"actor_loss": surrogate_loss, "critic_loss": critic_loss})


    # evaluate training steps
    


        
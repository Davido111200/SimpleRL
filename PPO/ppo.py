import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as distributions

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import namedtuple, deque
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.env_util import make_vec_env


from torch.distributions import Categorical
import wandb  

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

class ActorCritic(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(n_inputs, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, n_outputs)
        )
        self.critic = nn.Sequential(
            nn.Linear(n_inputs, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    
    def get_value(self, x):
        return self.critic(x)
    
    def get_action_and_probs_and_values(self, env, x, action=None):
        mean = self.actor(x)

        cov_var = torch.full(size=(env.action_space.shape[0],), fill_value=0.5).to(device)
        cov_mat = torch.diag(cov_var).to(device)
        probs = distributions.MultivariateNormal(loc=mean, covariance_matrix=cov_mat.to(device))
        if action is None:
            action = probs.sample()
        values = self.critic(x)
        return action.detach().cpu().numpy(), probs.log_prob(action), probs.entropy(), values
    
class PPO():
    def __init__(self, n_epochs, env_name, n_envs, n_step_per_batch, batch_size, epsilon, vf_coef, ent_coef) -> None:
        super().__init__()
        self.env = make_vec_env(env_name, n_envs=n_envs)
        self.actor_critic = ActorCritic(self.env.observation_space.shape[0], self.env.action_space.shape[0]).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=1e-5)
        self.n_step_per_batch = n_step_per_batch
        self.n_actors = n_envs
        self.n_epochs = n_epochs
        self.batch_size = int(n_envs * self.n_step_per_batch)
        self.update_epochs = 4 # the k-epoch 
        self.n_updates = self.n_epochs // self.batch_size
        self.cur_highest_reward = -1000000

        # hyperparameters
        self.lambd = 0.95
        self.gamma = 0.99
        self.epsilon = epsilon
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

        # storage setup
        self.states = torch.zeros((self.n_step_per_batch, n_envs, self.env.observation_space.shape[0])).to(device)
        self.actions = torch.zeros((self.n_step_per_batch, n_envs, self.env.action_space.shape[0])).to(device)
        self.log_probs = torch.zeros((self.n_step_per_batch, n_envs)).to(device)
        self.dones = torch.zeros((self.n_step_per_batch, n_envs)).to(device)
        self.values = torch.zeros((self.n_step_per_batch, n_envs)).to(device)
        self.rewards = torch.zeros((self.n_step_per_batch, n_envs)).to(device)
        self.next_states = torch.zeros((self.n_step_per_batch, n_envs, self.env.observation_space.shape[0])).to(device)


    def rollout(self):
        """
        Collect data with multiple agents.
        self.: number of timesteps to collect data for each agent
        """
        states = self.env.reset()

        # collecting T data for each agent 
        # T= self.n_step_per_batch here
        rews = []
        rew_agent = [0 for _ in range(self.n_actors)]
        for step in range(0, self.n_step_per_batch):
            with torch.no_grad():
                # run old policy in env for T timesteps
                actions, log_probs, entropys, state_values = self.actor_critic.get_action_and_probs_and_values(self.env, torch.from_numpy(states).float().to(device), None)

            # step
            next_states, rewards, dones, _ = self.env.step(actions)

            self.states[step] = torch.from_numpy(states).float().to(device)
            self.dones[step] = torch.from_numpy(dones).float().to(device)
            self.actions[step] = torch.from_numpy(actions).float().to(device)
            self.log_probs[step] = log_probs
            self.rewards[step] = torch.from_numpy(rewards).float().to(device)
            self.values[step] = state_values.flatten()
            
            states = next_states
            rew_agent += rewards
            for idx, done in enumerate(dones):
                if done:
                    rews.append(rew_agent[idx])
                    rew_agent[idx] = 0
            
        # check if any element of rews is invalid
        if np.isnan(np.mean(np.array(rews))):
            print(np.array(rews))
            quit()

        wandb.log({"reward": np.mean(np.array(rews))})
        # finished collecting T samples for each agent
        # save the reward to self.temp_rew

        with torch.no_grad():
            next_state_values = self.actor_critic.get_value(torch.from_numpy(next_states).float().to(device)).reshape(1, -1)
            self.advantages = torch.zeros_like(self.rewards).to(device)
            lastgaelam = 0

            for t in reversed(range(self.n_step_per_batch)):
                if t == self.n_step_per_batch - 1:
                    next_non_terminal = 1.0 - self.dones[-1]
                    next_state_value = next_state_values
                else:
                    next_non_terminal = 1.0 - self.dones[t + 1]
                    next_state_value = self.values[t + 1]

                delta = self.rewards[t] + self.gamma * next_state_value * next_non_terminal - self.values[t]
                self.advantages[t] = lastgaelam = delta + self.gamma * self.lambd * next_non_terminal * lastgaelam
            self.returns = self.advantages + self.values

        # flatten the batch and then return
        b_states = self.states.reshape(-1, self.env.observation_space.shape[0])
        b_actions = self.actions.reshape(-1, self.env.action_space.shape[0])
        b_log_probs = self.log_probs.reshape(-1)
        b_advantages = self.advantages.reshape(-1)
        b_returns = self.returns.reshape(-1)
        b_values = self.values.reshape(-1)

        return b_states, b_actions, b_log_probs, b_advantages, b_returns, b_values
    
    def learn(self):
        b_indicies = np.arange(self.batch_size)
        num_mini_batch = 4
        mini_batch_size = self.batch_size // num_mini_batch

        for epoch in range(self.n_updates):
            # collect batch data
            b_states, b_actions, b_log_probs, b_advantages, b_returns, b_values = self.rollout()


            for update_epoch in range(self.update_epochs):
                np.random.shuffle(b_indicies)
                for start in range(0, self.batch_size, mini_batch_size):
                    end = start + mini_batch_size
                    mb_inds = b_indicies[start:end]

                    # minibatch advantage
                    mb_advantages = b_advantages[mb_inds]
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # calculate the log_probs for the current policy
                    _, current_log_probs, entropy, old_values = self.actor_critic.get_action_and_probs_and_values(self.env, b_states[mb_inds], b_actions[mb_inds])

                    # calculate the ratio
                    ratio = torch.exp(current_log_probs - b_log_probs[mb_inds])
                    # calculate the surrogate loss
                    surrogate_loss_1 = ratio * mb_advantages
                    surrogate_loss_2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * mb_advantages

                    surrogate_loss = (-torch.min(surrogate_loss_1, surrogate_loss_2)).mean()

                    # calculate the value loss
                    target_values = b_returns[mb_inds]
                    criterion = nn.MSELoss()
                    value_loss = criterion(old_values.squeeze(), target_values)

                    loss = surrogate_loss + self.vf_coef * value_loss - self.ent_coef * entropy.mean()

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                    self.optimizer.step()

                    # log the loss to wandb
                    wandb.log({"loss": loss.item()})
                    wandb.log({"surrogate_loss": surrogate_loss.item()})
                    wandb.log({"value_loss": value_loss.item()})







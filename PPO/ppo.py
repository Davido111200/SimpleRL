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
from stable_baselines3.common.buffers import RolloutBuffer

from torch.distributions import Categorical
import wandb  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

class Actor(nn.Module):
    def __init__(self, n_state, n_action) -> None:
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_state, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, n_action)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = self.fc4(x)
        return x

class Critic(nn.Module):
    def __init__(self, n_state) -> None:
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_state, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = self.fc4(x)
        return x

class PPO:
    def __init__(self, env, batch_size, epsilon, n_envs) -> None:
        self.actor = Actor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
        self.critic = Critic(env.observation_space.shape[0]).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.005)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.005)

        # hyperparameters
        # ent_coef: entropy coefficient for joint loss calculation
        # vf_coef: value function coefficient for joint loss calculation
        self.ent_coef = 0
        self.vf_coef = 0.5

        self.env = env 
        self.n_envs = n_envs
        self.discount_factor = 0.95
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.n_update_per_epoch = 5
        self.n_step = self.batch_size * self.n_envs

        # init the rollout buffer
        self.rollout_buffer = RolloutBuffer(self.batch_size * self.n_envs, self.env.observation_space, self.env.action_space, device=device, n_envs=self.n_envs,
                                            gamma=self.discount_factor)
        self.rollout_buffer_size = self.batch_size * self.n_envs
        self.rollout_buffer.entropy =np.zeros((self.rollout_buffer_size, self.n_envs), dtype=np.float32)

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

        state_value = self.critic(state)

        dist = distributions.MultivariateNormal(loc=mean, covariance_matrix=self.cov_mat)

        action = dist.sample().to(device)

        log_prob = dist.log_prob(action)

        entropy = dist.entropy().mean()

        return action.detach().cpu().numpy(), log_prob.detach(), entropy.cpu().numpy(), state_value

    def compute_rtg(self, rewards, discount_factor):
        """
        Return rewards-to-go for each time-step
        We go reverse to get the full trajectory rtgs
        """
        rtgs = []
        
        for ep_rew in reversed(rewards):
            discounted_reward = 0  # Initialize with a float

            for rew in reversed(ep_rew):
                discounted_reward = rew + discount_factor * discounted_reward
                rtgs.insert(0, discounted_reward)
        
        rtgs = torch.tensor(rtgs, dtype=torch.float32, device=device)
        
        return rtgs

    def evaluate_actions(self, states, actions):
        V = self.critic(states)

        mean = self.actor(states)
        dist = distributions.MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(actions)

        return V, log_probs

    def sample_with_indices(self, batch_size: int, env=None):
        """
        Sample the buffer and return both the sampled data and their corresponding indices.
        :param batch_size: Number of elements to sample
        :param env: Associated gym VecEnv to normalize observations/rewards when sampling
        :return: Tuple containing (batch_indices, batch_states, batch_actions, batch_values, batch_log_probs,
                batch_advantages, batch_returns)
        """
        upper_bound = self.rollout_buffer_size if self.rollout_buffer.full else self.rollout_buffer.pos
        batch_indices = np.random.randint(0, upper_bound, size=batch_size)
        batch_data = self.rollout_buffer._get_samples(batch_indices, env=env)

        return batch_indices, batch_data

    def rollout(self):
        """
        Get the batch data from self.rollout_buffer
        """

        # reset the rollout buffer
        self.rollout_buffer.reset()

        state = self.env.reset()
        dones = [False] * self.n_envs

        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        rews = []
        rew_agent = [0] * self.n_envs
        for step in range(self.n_step):
            action, log_prob, entropy, state_value = self.get_action(state)

            next_state, reward, dones, infos = self.env.step(action)
            next_state = torch.as_tensor(next_state, dtype=torch.float32, device=device)
            self.rollout_buffer.compute_returns_and_advantage(last_values=state_value.detach(), dones=dones)
            self.rollout_buffer.add(obs=state, action=action, reward=reward, episode_start=dones, log_prob=log_prob,
                                    value=state_value.detach())
            
            state = next_state

            
        batch_states, batch_actions, batch_values, batch_log_probs, batch_advantages, batch_returns = self.rollout_buffer.sample(batch_size=self.n_step, env=self.env)

        # keep track of the rewards and report to wandb
        wandb.log({"reward": torch.mean(batch_returns).item()})

        return batch_states, batch_actions, batch_values, batch_log_probs, batch_advantages, batch_returns

    def learn(self, n_epochs):
        t_so_far = 0
        while t_so_far < n_epochs:
            # rollout to take the data
            batch_states, batch_actions, batch_values, batch_log_probs, batch_advantages, batch_returns = self.rollout()
            
            t_so_far += len(batch_returns)
            
            # log t_so_far to wandb
            wandb.log({"t_so_far": t_so_far})

            # compute advantage
            A_k = batch_advantages

            # normalize advantage
            A_k = (A_k - A_k.mean()) / (A_k.std(unbiased=False) + 1e-10)

            # compute the loss
            # actor loss
            # the new policy probability = taking the probs with current actor for the action considering
            # compute the log probability for current policy
            for learning_step in range(self.n_update_per_epoch):
                V, current_log_action = self.evaluate_actions(batch_states, batch_actions)

                # the old log probability is calculated in rollout
                # now we calculate the ratio
                ratio = torch.exp(current_log_action.view(len(batch_log_probs)) - batch_log_probs)

                surrogate_loss_1 = ratio * A_k
                surrogate_loss_2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * A_k

                surrogate_loss = (- torch.min(surrogate_loss_1, surrogate_loss_2)).mean()
                # self.actor_optimizer.zero_grad()
                # surrogate_loss.backward()
                # self.actor_optimizer.step()

                critic_loss = nn.MSELoss()(V.flatten(), batch_advantages + batch_values)
                actor_loss = surrogate_loss

                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                # Clip gradient norms
                # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.critic_optimizer.step()


                wandb.log({"actor_loss": surrogate_loss, "critic_loss": critic_loss})


        
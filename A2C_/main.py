import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils as nn_utils

import numpy as np
import random
import gymnasium as gym
import argparse
import matplotlib.pyplot as plt
from collections import deque
import wandb
import os
import tempfile
import logging
from wandb import env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

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
        x = self.fc2(x)
        return x

def plot_test_scores(test_scores, filename):
    "Plot test scores"
    plt.clf()
    plt.plot(test_scores)
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.title('Test Scores')
    plt.savefig(filename)
    plt.show()

def plot_reward_trials_with_variance(trial_scores, filename, blurred_variance_factor=0.3):
    """
    Plots multiple reward trials with an average line and blurred variance.

    Args:
        trial_scores (list of arrays): List containing arrays of trial scores for each trial.
        blurred_variance_factor (float): Factor controlling the amount of blurring for variance.

    Returns:
        None
    """
    num_trials = len(trial_scores)

    # Calculate the average and variance
    average_scores = np.mean(trial_scores, axis=0)
    blurred_variance = np.std(trial_scores, axis=0) * blurred_variance_factor

    # Plot the average line
    plt.plot(average_scores, label='Average', color='blue')

    # Plot the blurred variance area
    plt.fill_between(range(len(average_scores)), average_scores - blurred_variance, average_scores + blurred_variance, alpha=0.3, color='blue')

    # Plot individual trial scores
    for i in range(num_trials):
        plt.plot(trial_scores[i], color='gray', alpha=0.3)

    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.title('Reward Trials with Average and Blurred Variance')
    plt.savefig(filename)
    plt.legend()
    plt.show()

def main(n_epochs, max_ts, n_trials, test_epochs):
    wandb.init(project='A2C_CartPole-v1', config={'n_epochs': n_epochs, 'max_ts': max_ts, 'n_trials': n_trials})   
    trial_scores = []
    env = gym.make("CartPole-v1")

    for trial in range(n_trials):
        scores = []
        actor = Actor(env.observation_space.shape[0], env.action_space.n).to(device)
        actor_optimizer = optim.Adam(actor.parameters(), lr=0.0001)
        critic = Critic(env.observation_space.shape[0]).to(device)
        critic_optimizer = optim.Adam(critic.parameters(), lr=0.0001)

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

                done = terminated or truncated

                
                log_prob_action = action_dist.log_prob(action).to(device)

                current_state_value = critic(state)

                if done:
                    td_error = reward - current_state_value.item()
                else:
                    next_state = torch.from_numpy(next_state).float().to(device)
                    next_state_value = critic(next_state)
                    td_error = reward + gamma * next_state_value.item() - current_state_value.item()
                
                # Create detached tensors for gradient computation
                reward_tensor = torch.tensor([reward], dtype=torch.float32, device=device)
                next_state_value_tensor = torch.tensor([next_state_value.item()], dtype=torch.float32, device=device) if not done else torch.tensor([0.0], dtype=torch.float32, device=device)

                # critic update
                criterion = nn.MSELoss()
                loss_critic = criterion(current_state_value, reward_tensor + gamma * next_state_value_tensor)
                critic_optimizer.zero_grad()
                loss_critic.backward()
                critic_optimizer.step()

                # actor update
                loss_actor = - td_error * step_size_actor * log_prob_action * I
                actor_optimizer.zero_grad()
                loss_actor.backward()
                actor_optimizer.step()

                state = next_state
                I *= gamma

                if done:
                    break
            
            running_reward = running_reward * 0.99 + ts * 0.01
            scores.append(running_reward)
            wandb.log({'Episode Length': ts, 'Running Reward': running_reward, 'Trial': trial, 'Epoch': epoch})

            if epoch % 100 == 0:
                print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                    epoch, ts, running_reward))
            if running_reward > env.spec.reward_threshold:
                print("Solved! Running reward is now {} and "
                    "the last episode runs to {} time steps!".format(running_reward, ts))
        trial_scores.append(scores)
        
        # save the weights of each trial
        torch.save(actor.state_dict(), "/home/s223540177/dai/SimpleRL/A2C/weights/actor_trial{}.pth".format(trial+1))

    for trial, scores in enumerate(trial_scores):
        wandb.log({"Trial {}".format(trial+1): scores[-1]})
    

    plot_reward_trials_with_variance(trial_scores, filename="/home/s223540177/dai/SimpleRL/A2C/figs/training_plot.png", blurred_variance_factor=0.3)

    # evaluate the agent with the mean of the weights

    actor.load_state_dict(torch.load("/home/s223540177/dai/SimpleRL/A2C/weights/weights_trial_{}.pth".format(np.argmax(np.array(trial_scores)[:, -1]))))

    test_scores = []
    for te in range(test_epochs):
        state, _ = env.reset()
        done = False
        ts = 0
        while not done:
            state = torch.from_numpy(state).float().to(device)
            action_probs = actor(state)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()

            next_state, reward, terminated, truncated, _ = env.step(action.item())

            done = terminated or truncated

            state = next_state
            ts += 1

        test_scores.append(ts)
        print("Test Epoch: {}\tTest Score: {}".format(te, ts))

    # plot the reward trials
    plot_test_scores(test_scores, filename="/home/s223540177/dai/SimpleRL/A2C/figs/test_plot.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Actor-Critic')
    parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--max_ts', type=int, default=1000, help='max time steps')
    parser.add_argument('--n_trials', type=int, default=5, help='number of trials')
    parser.add_argument('--test_epochs', type=int, default=100, help='number of test epochs')
    args = parser.parse_args()
    main(args.n_epochs, args.max_ts, args.n_trials, args.test_epochs)
    wandb.finish()
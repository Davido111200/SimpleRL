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

def plot_training_rewards_by_time_step(rewards, filename):
    """
    Plots the reward for each time step in the training process.

    Args:
        rewards (list of floats): List containing the reward for each time step in the training process.

    Returns:
        None
    """
    plt.clf()
    plt.plot(rewards)
    plt.xlabel('Time Step')
    plt.ylabel('Reward')
    plt.title('Reward by Time Step')
    plt.savefig(filename)
    plt.show()

def main(n_epochs, max_ts, n_trials, test_epochs):
    pi = policy(env.observation_space.shape[0], env.action_space.n)
    optimizer = optim.Adam(pi.parameters(), lr=0.01)
    gamma = 0.99

    trial_scores = []

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

                next_state, reward, terminated, truncated, _ = env.step(action.item())

                done = terminated or truncated
                
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
            returns = (returns - returns.mean()) / (returns.std() + 1e-7)

            for lp, Gt in zip(log_probs, returns):
                policy_loss.append(-lp * Gt)
            
            optimizer.zero_grad()
            policy_loss = sum(policy_loss)
            policy_loss.backward()
            optimizer.step()


            running_reward = running_reward * 0.99 + ts * 0.01
            scores.append(running_reward)

            if epoch % 100 == 0:
                print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                    epoch, ts, running_reward))
            if running_reward > env.spec.reward_threshold:
                print("Solved! Running reward is now {} and "
                    "the last episode runs to {} time steps!".format(running_reward, ts))
        trial_scores.append(scores)
        # save the weights of each trial
        torch.save(pi.state_dict(), "/home/s223540177/dai/SimpleRL/REINFORCE/weights/weights_trial_{}.pth".format(trial))
        
    plot_reward_trials_with_variance(trial_scores, filename="/home/s223540177/dai/SimpleRL/REINFORCE/figs/training_plot.png", blurred_variance_factor=0.3)

    # evaluate the agent with the mean of the weights
    pi.load_state_dict(torch.load("/home/s223540177/dai/SimpleRL/REINFORCE/weights/weights_trial_{}.pth".format(np.argmax(np.array(trial_scores)[:, -1]))))

    test_scores = []
    for te in range(test_epochs):
        state, _ = env.reset()
        done = False
        ts = 0
        while not done:
            state = torch.from_numpy(state).float().to(device)
            action_probs = pi(state)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()

            next_state, reward, terminated, truncated, _ = env.step(action.item())

            done = terminated or truncated

            state = next_state
            ts += 1

        test_scores.append(ts)
        print("Test Epoch: {}\tTest Score: {}".format(te, ts))

    # plot the reward trials
    plot_test_scores(test_scores, filename="/home/s223540177/dai/SimpleRL/REINFORCE/figs/test_plot.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--max_ts", type=int, default=1000)
    parser.add_argument("--n_trials", type=int, default=5)
    parser.add_argument("--test_epochs", type=int, default=100)
    args = parser.parse_args()
    main(args.n_epochs, args.max_ts, args.n_trials, args.test_epochs)
    
    


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from short_corridor import short_env
import argparse
import random
import tqdm
import matplotlib.pyplot as plt
import matplotlib

device = torch.device("cpu")

env = short_env(n_states=4)


class policy_network(nn.Module):
    def __init__(self, n_inputs, hidden, n_outputs) -> None:
        super(policy_network, self).__init__()
        self.n_inputs = n_inputs
        self.hidden = hidden
        self.n_outputs = n_outputs
        self.fc1 = nn.Linear(self.n_inputs, self.hidden, device=device)
        self.fc2 = nn.Linear(self.hidden, self.n_outputs, device=device)

    def forward(self, x):
        x = self.state_to_tensor(x)
        x = F.relu(self.fc1(x))
        action_probs = F.softmax(self.fc2(x), dim=0)
        selected_action = torch.multinomial(action_probs, num_samples=1)
        selected_action_prob = action_probs[selected_action]
        selected_action_log_prob = torch.log(selected_action_prob)
        return selected_action, selected_action_log_prob

    def state_to_tensor(self, state):
        # one-hot
        dummy = torch.zeros(size=(env.n_states, 1))
        dummy[int(state)] = 1
        dummy = dummy.flatten()
        return dummy
    
    def sample(self):
        return np.random.randint(low=0, high=env.n_actions)

def main(n_epochs, n_runs):
    total_reward_per_run = []

    for run in range(n_runs):
        policy = policy_network(n_inputs=env.n_states, hidden=128, n_outputs=env.n_actions)
        optimizer = optim.AdamW(policy.parameters())

        ALPHA = 2* 1e-13
        GAMMA = 0.9

        total_reward = []
        for epoch in range(n_epochs):
            states = []
            actions = []
            rewards = [] # reward counts from 1, while the 2 aboves are from 0
            lp = []
            state, terminated = env.reset()
            states.append(state) # first state

            while not terminated:
                # generate a whole trajectory
                action, selected_action_log_prob = policy(state)
                
                next_state, reward, terminated = env.step(action)
                
                actions.append(action)
                states.append(next_state)
                rewards.append(reward)
                lp.append(selected_action_log_prob)

                state = next_state 

            discounted_reward = []
            for t in range(len(rewards)):
                G = 0
                for k in range(t, len(rewards)):
                    G += (GAMMA ** k) * rewards[k]
                discounted_reward.append(G * GAMMA ** t)

            discounted_reward_tensor = torch.tensor(discounted_reward, dtype=torch.float32, device=device)

            # save the reward per episode (normalize) for plotting
            total_reward.append(torch.mean(discounted_reward_tensor).item())

            log_probs = torch.stack(lp).flatten()
            
            # there is a negative sign here as the algorithm requires gradient ascent
            policy_gradient = -(ALPHA * log_probs * discounted_reward_tensor).mean()
            
            policy.zero_grad()
            policy_gradient.backward()
            optimizer.step()

        total_reward_per_run.append(total_reward)
    result_reward = [sum(x) for x in zip(*total_reward_per_run)]

    def plot(rewards):
        window_size = 100

        # Calculate the moving average using a convolution with a uniform window of size window_size
        moving_average = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')

        # Calculate the average of the moving averages at each time step
        average_moving_average = np.mean(moving_average)

        # Create an array of time steps (assuming the data points are evenly spaced)
        time_steps = np.arange(window_size // 2, window_size // 2 + len(moving_average))

        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, moving_average, label=f"Moving Average (window_size={window_size})")
        plt.axhline(average_moving_average, color='red', linestyle='dashed', label='Average of Moving Averages')
        plt.xlabel('Time')
        plt.ylabel('Moving Average')
        plt.legend()
        plt.title('Average of Moving Averages Over Time')
        plt.grid(True)
        plt.show()
    
    plot(total_reward_per_run)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=400)
    parser.add_argument("--n_runs", type=int, default=100)

    arguments = parser.parse_args()

    main(arguments.n_epochs, arguments.n_runs)
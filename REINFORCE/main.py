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
        selected_action = self.select_action(action_probs=action_probs, eps=0.1)
        selected_action_prob = action_probs[selected_action]
        selected_action_log_prob = torch.log(selected_action_prob)
        return selected_action, selected_action_log_prob
        
    def select_action(self, action_probs, eps=0.1):
        if random.random() < eps:
            return np.random.randint(low=0, high=env.n_actions)
        else:
            return torch.argmax(action_probs).detach().numpy()

    def state_to_tensor(self, state):
        # one-hot
        dummy = torch.zeros(size=(env.n_states, 1))
        dummy[int(state)] = 1
        dummy = dummy.flatten()
        return dummy
    
    def sample(self):
        return np.random.randint(low=0, high=env.n_actions)

def main(n_epochs):
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
        for idx, r in enumerate(rewards):
            G = 0 
            for j in range(idx, len(rewards)):
                G += GAMMA ** (j-idx) * rewards[j]
                
            discounted_reward.append(G)
        
        discounted_reward_tensor = torch.tensor(discounted_reward, dtype=torch.float32, device=device)

        # save the reward per episode (normalize) for plotting
        total_reward.append(torch.mean(discounted_reward_tensor).item())
        

        log_probs = torch.stack(lp)

        assert log_probs.shape == discounted_reward_tensor.shape, "hahant"

        # there is a negative sign here as the algorithm requires gradient ascent
        policy_gradient = -(ALPHA * log_probs * discounted_reward_tensor).mean()
        
        policy.zero_grad()
        policy_gradient.backward()
        optimizer.step()

    def plot():
        x = np.arange(0, n_epochs)
        y = np.array(total_reward)

        plt.plot(x, y)
        plt.show()

    plot()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=400)

    arguments = parser.parse_args()

    main(arguments.n_epochs)
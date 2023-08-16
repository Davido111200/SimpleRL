# episodic case
import gymnasium as gym
import random
import numpy as np
from env import environment
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class Actor(nn.Module):
    def __init__(self, n_inputs, n_hiddens, n_outputs) -> None:
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_inputs, n_hiddens, dtype=torch.float32, device=device)
        self.fc2 = nn.Linear(n_hiddens, n_outputs, dtype=torch.float32, device=device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_probs = F.softmax(self.fc2(x), dim=0)
        selected_action = torch.multinomial(action_probs, num_samples=1)
        return selected_action, action_probs
    
class Critic(nn.Module):
    def __init__(self, n_inputs, n_hiddens) -> None:
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_inputs, n_hiddens, dtype=torch.float32, device=device)
        self.fc2 = nn.Linear(n_hiddens, 1, dtype=torch.float32, device=device)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
def main(n_epochs, max_ts):
    env = environment().env

    ### PARAMS
    GAMMA = 0.9
    alph_thet = 0.7
    alph_w = 0.7
    
    torch.autograd.set_detect_anomaly(True)

    actor = Actor(env.observation_space.shape[0], 128, env.action_space.n)
    critic = Critic(env.observation_space.shape[0], 128)

    actor_optimizer = optim.AdamW(actor.parameters())
    critic_optimizer = optim.AdamW(critic.parameters())

    for epoch in range(n_epochs):
        state, _ = env.reset()
        terminated = False

        reward_epoch = 0
        I = 1

        for ts in range(max_ts):
            state = torch.as_tensor(state, dtype=torch.float32, device=device)
            action, action_probs = actor(state)
            next_state, reward, terminated, truncated, _ = env.step(action.item())

            # save reward for measuring performance
            reward_epoch += reward

            next_state = torch.as_tensor(next_state, dtype=torch.float32, device=device)
            next_state_val = critic(next_state) if not terminated else 0
            current_state_val = critic(state)

            # calculate value function loss with MSE
            val_loss = F.mse_loss(torch.tensor(reward+ GAMMA * next_state_val, device=device), current_state_val) 
            val_loss *= alph_w

            # calculate policy loss
            advantage = reward + GAMMA  * next_state_val - current_state_val.item()
            policy_loss = - torch.log(action_probs)[action] * advantage * alph_thet
            policy_loss *= I

            # backprop for policy
            actor_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            actor_optimizer.step()

            # backprop for value function
            critic_optimizer.zero_grad()
            val_loss.backward()
            critic_optimizer.step()

            if terminated:
                break

            I *= GAMMA
            state = next_state

        if epoch % 10 == 0:
            print("eps {}: {}".format(epoch, reward_epoch))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--max_steps', type=int, default=10000)
    args = parser.parse_args()
    main(args.n_epochs, args.max_steps)
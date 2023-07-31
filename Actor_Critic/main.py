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

device = torch.device('cpu')

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
    
def main(n_epochs):
    env = environment().env

    ### PARAMS
    GAMMA = 0.9
    lamb_thet = 0.7
    lamb_w = 0.7
    alph_thet = 0.3
    alph_w = 0.3

    torch.autograd.set_detect_anomaly(True)

    actor = Actor(env.observation_space.shape[0], 128, env.action_space.n)
    critic = Critic(env.observation_space.shape[0], 128)

    criterion = nn.MSELoss()

    optimizer_actor = optim.SGD(params=actor.parameters(), lr=0.001)
    optimizer_critic = optim.SGD(params=critic.parameters(), lr=0.001)

    for epoch in range(n_epochs):
        state, _ = env.reset()
        terminated = False
        z_thet = [torch.zeros_like(param, dtype=torch.float32) for param in actor.parameters()]
        z_w = [torch.zeros_like(param, dtype=torch.float32) for param in critic.parameters()]
        I = 1
        while not terminated:
            state = torch.as_tensor(state, dtype=torch.float32, device=device)
            action, action_probs = actor(state)
            next_state, reward, terminated, truncated, _ = env.step(action.item())

            next_state = torch.as_tensor(next_state, dtype=torch.float32, device=device)
            target = reward + critic(next_state)
            estimated_state_val = critic(state)

            td_error = target.detach() - estimated_state_val

            ## trace w - update
            loss_w = (target.detach() - estimated_state_val)
            critic.zero_grad()
            loss_w.backward()

            # calculate and save the gradients
            gradients = []
            for param in critic.parameters():
                gradients.append(param.grad.data.clone())

            # updating z_w
            z_w = [GAMMA * alph_w * i for i in z_w]
            z_w += gradients

            # updating w
            z_w_updates = [alph_w * td_error.item() * i for i in z_w]
            assert len(z_w_updates) == len(z_w)
            w_new_params = list(critic.parameters()) + z_w_updates
            with torch.no_grad():
                for idx, p in enumerate(critic.parameters()):
                    p.copy_(w_new_params[idx])
            
            quit()
            # trace theta - update
            action_log_probs = torch.log(action_probs).requires_grad()
            print(action_log_probs)
            print(target)
            gradients_theta = - target.detach() * action_log_probs
            print(gradients_theta)
            quit()
            
            


            







if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_epochs', type=int, default=1)

    args = parser.parse_args()
    main(args.n_epochs)
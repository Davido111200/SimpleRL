import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import random
import math
import argparse
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib

from sarsa import SARSA

def plot(rew_list):
    plt.figure(figsize=(12, 5))
    plt.xlabel("Run number")
    plt.ylabel("Outcome")
    ax = plt.gca()
    ax.set_facecolor('#efeeea')
    plt.bar(range(len(rew_list)), rew_list, color="#0A047A", width=1.0)
    plt.show()

def main(n_epochs):
    env = gym.make('FrozenLake-v1', map_name="4x4", render_mode='rgb_array')
    qtable = np.zeros((env.observation_space.n, env.action_space.n))

    # HYPERS
    eps=0.1
    GAMMA=0.9
    ALPHA = 0.4

    print("q-table: before", qtable)

    def epsilon_greedy(state, q_table):
        temp = random.random()
        if temp < eps or np.max(q_table[state]) == 0:
            return env.action_space.sample()
        else:
            return np.argmax(q_table[state])


    # SARSA first
    rews = []

    for epoch in range(n_epochs):
        state, _ = env.reset()

        # choose action based on epsilon-greedy - which is the current policy
        action = epsilon_greedy(state, qtable)
        terminated = False
        n_success = 0
        
        while not terminated:
            next_state, reward, terminated, t, info = env.step(action)
            
            # SARSA target policy is behavior policy
            next_action = epsilon_greedy(next_state, qtable)

            target_action_value = reward + GAMMA * qtable[next_state, next_action]

            # update
            qtable[state, action] = qtable[state, action] + \
                                ALPHA * (target_action_value - qtable[state, action])

            if reward:
                n_success += 1

            state = next_state
            action = next_action
        rews.append(n_success)

    print(rews)
    print('q_table after', qtable)
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int)

    args = parser.parse_args()
    main(args.n_epochs)

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
import sys
import json


def plot(rew_list):
    plt.figure(figsize=(12, 5))
    plt.xlabel("Run number")
    plt.ylabel("Outcome")
    ax = plt.gca()
    ax.set_facecolor('#efeeea')
    plt.bar(range(len(rew_list)), rew_list, color="#0A047A", width=1.0)
    plt.show()

def save_output(path, rewards):
    with open("/home/s223540177/dai/SimpleRL/n_step_sarsa/output.txt", 'w') as f: 
        for key, value in rewards.items(): 
            f.write('%s:%s\n' % (key, value))

def main(n_epochs, max_ts, n):
    env = gym.make('FrozenLake-v1', map_name="4x4", render_mode='rgb_array')

    # HYPERS
    eps=0.1
    GAMMA=0.9
    ALPHA = 0.1

    def epsilon_greedy(state, q_table):
        temp = random.random()
        if temp < eps or np.max(q_table[state]) == 0:
            return env.action_space.sample()
        else:
            return np.argmax(q_table[state])

    actions = []
    rewards = []
    states = []
    list_n = []
    rews_n = {}

    for i in range(1, n+1):
        list_n.append(i**2)

    for n in list_n:
        qtable = np.zeros((env.observation_space.n, env.action_space.n))
        
        print('working on {}'.format(n))
        print("q-table: before", qtable)

        rews = []

        for epoch in range(n_epochs):
            state, _ = env.reset()
            
            # select and store an action A_0
            action = epsilon_greedy(state, qtable)
            actions.append(action)

            terminated = False
            T = max_ts
            epoch_reward = 0

            for t in range(T): 
                if t < T:
                    next_state, reward, terminated, truncated, _ = env.step(action)

                    epoch_reward += reward

                    # store next reward and next state
                    rewards.append(reward)
                    states.append(next_state)

                    # possible error
                    if terminated:
                        T = t + 1
                    else:
                        # select and store next action
                        next_action = epsilon_greedy(next_state, qtable)
                        actions.append(next_action)
                
                # set tau variable for 
                tau = t - n + 1

                if tau >= 0:
                    G = 0
                    for idx, ret in enumerate(rewards):
                        G += GAMMA**idx * ret
                    if tau + n < T:
                        G += GAMMA ** n * qtable[states[-1], actions[-1]]
                    
                    # update q value for tau
                    qtable[states[tau], actions[tau]] = qtable[states[tau], actions[tau]] \
                    + ALPHA * (G - qtable[states[tau], actions[tau]])
                
                # terminnation ?
                if tau == T - 1:
                    break
            rews.append(epoch_reward)

        print("q-table: after", qtable)
        rews_n[n] = rews
    save_output('SimpleRL/n_step_sarsa/output.txt', rews_n)

                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int)
    parser.add_argument('--max_ts', type=int)
    parser.add_argument('--n', type=int)

    args = parser.parse_args()
    main(args.n_epochs, args.max_ts, args.n)

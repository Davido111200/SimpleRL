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
    ALPHA = 0.1

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

        terminated = False
        n_success = 0
        
        while not terminated:
            action = epsilon_greedy(state, qtable)

            next_state, reward, terminated, t, info = env.step(action)
            
            target_action_value = reward + GAMMA * np.max(qtable[next_state, :])

            # update
            qtable[state, action] = qtable[state, action] + \
                                ALPHA * (target_action_value - qtable[state, action])

            if reward:
                n_success += 1

            state = next_state
        rews.append(n_success)

    print(rews)
    print('q_table after', qtable)

    episodes = 100
    nb_success = 0

    # Evaluation
    for _ in range(100):
        state, _ = env.reset()
        done = False
        
        # Until the agent gets stuck or reaches the goal, keep training it
        while not done:
            # Choose the action with the highest value in the current state
            if np.max(qtable[state, :]) > 0:
                action = np.argmax(qtable[state, :])

            # If there's no best action (only zeros), take a random one
            else:
                action = env.action_space.sample()
                
            # Implement this action and move the agent in the desired direction
            new_state, reward, done, t, info = env.step(action)

            # Update our current state
            state = new_state

            # When we get a reward, it means we solved the game
            nb_success += reward

    # Let's check our success rate!
    print (f"Success rate = {nb_success/episodes*100}%")
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int)

    args = parser.parse_args()
    main(args.n_epochs)

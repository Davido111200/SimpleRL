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
import shutil

def plot(rew_list):
    plt.figure(figsize=(12, 5))
    plt.xlabel("Run number")
    plt.ylabel("Outcome")
    ax = plt.gca()
    ax.set_facecolor('#efeeea')
    plt.bar(range(len(rew_list)), rew_list, color="#0A047A", width=1.0)
    plt.show()

def main(n_epochs, n_trials):
    env = gym.make('Taxi-v3', render_mode='rgb_array')
    qtable = np.zeros((env.observation_space.n, env.action_space.n))

    # HYPERS
    eps=0.1
    GAMMA=0.9
    ALPHA = 0.1
    init_rewards = -99999999

    def epsilon_greedy(state, q_table):
        temp = random.random()
        if temp < eps or np.max(q_table[state]) == 0:
            return env.action_space.sample()
        else:
            return np.argmax(q_table[state])

    rews = []

    for trial in range(n_trials):
        trial_results = []

        for epoch in range(n_epochs):
            state, _ = env.reset()

            terminated = False
            epoch_reward = 0
            
            while not terminated:
                action = epsilon_greedy(state, qtable)

                next_state, reward, terminated, t, info = env.step(action)
                
                target_action_value = reward + GAMMA * np.max(qtable[next_state, :])

                # update
                qtable[state, action] = qtable[state, action] + \
                                    ALPHA * (target_action_value - qtable[state, action])

                epoch_reward += reward

                state = next_state
            rews.append(np.mean(epoch_reward))
            
            # save the best qtable
            is_best = (np.mean(epoch_reward) > init_rewards)
            init_rewards = max(np.mean(epoch_reward), init_rewards)
            if is_best:
                eval_qtable = qtable
                print('Epoch', epoch, 'qtable saved')

        trial_results.append(rews)

    # Evaluation
    print('Start evaluation')
    for _ in range(100):
        state, _ = env.reset()
        done = False
        nb_success = 0

        
        # Until the agent gets stuck or reaches the goal, keep training it
        while not done:
            # Choose the action with the highest value in the current state
            if np.max(eval_qtable[state, :]) > 0:
                action = np.argmax(eval_qtable[state, :])

            # If there's no best action (only zeros), take a random one
            else:
                action = env.action_space.sample()
                
            # Implement this action and move the agent in the desired direction
            new_state, reward, done, t, info = env.step(action)

            # Update our current state
            state = new_state

            # When we get a reward, it means we solved the game
            nb_success += 1 if reward > 0 else 0

        # Let's check our success rate!
        print(f"Success rate: {nb_success / 100}%")    
    print('Mean reward', np.mean(trial_results))

    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int)
    parser.add_argument('--n_trials', type=int)

    args = parser.parse_args()
    main(args.n_epochs, args.n_trials)

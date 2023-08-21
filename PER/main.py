import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F

import yaml
import numpy as np
import random
from collections import deque, namedtuple
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
import math
from tree import SumTree
import gymnasium as gym

is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display


Transition = namedtuple("transition", ("state", "action", "reward", "next_state"))

# In fact, does using Cartpole env, with rewards at all time = 1, meaningful?
# This makes the return estimations meaning less? As reward at all time is 1, 
# returns depend on # of time steps


env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# HYPER PARAMS
N_EPS = 1000
GAMMA = 0.99
BATCH_SIZE = 128
BUFFER_SIZE = 10000
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 100
steps_done = 0
replay_period = 10
k = 128
ALPHA = 1
BETA = 1
TAU = 0.005
epsilon = 0.0 # this is the minimal prob of selecting -> avoid actions that are never selected

class PDQN(nn.Module):
    def __init__(self, n_obs, n_act) -> None:
        super(PDQN, self).__init__()
        self.fc1 = nn.Linear(n_obs, 256, device=device)
        self.fc2 = nn.Linear(256, n_act, device=device)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
# init network here
policy_net = PDQN(env.observation_space.shape[0] * env.observation_space.shape[1] *\
                  env.observation_space.shape[2] , env.action_space.n).to(device)
target_net = PDQN(env.observation_space.shape[0] * env.observation_space.shape[1] *\
                  env.observation_space.shape[2] , env.action_space.n).to(device)
theta = policy_net.parameters()

optimizer_policy = optim.AdamW(theta)

class PriortizedBuffer(object):
    def __init__(self, capacity) -> None:
        self.buffer = SumTree(size=capacity, epsilon=epsilon)
        self.counter = 0

    def push(self, *args):
        """
        This process is used for saving new experience, with the parameter used to rank it
        """
        # this step makes a transition to be added with priority = its current index

        if self.counter > BUFFER_SIZE - 1:
            # in this part, the buffer needs to update least relevant samples
            # first of all , we need to determine samples with the least probs
            smallest_priority_index, highest_priority_index = self.buffer.get_leaf_nodes_properties()
            self.buffer.replace_data(smallest_priority_index, Transition(*args))
            self.buffer.update(smallest_priority_index, highest_priority_index)
        else:
            self.buffer.add(self.counter, Transition(*args))

    def update_(self, indicies: list, new_priorities: list):
        assert len(indicies) == len(new_priorities), "update length error"
        for index, new_priority in zip(indicies, new_priorities):
            self.buffer.update(index, new_priority)

    def sample(self, k):
        p_total = self.buffer.nodes[0]
        ending_points = list(np.linspace(start=epsilon, stop=p_total, num=k))
        
        # now we sample 1 transition each from the corresponding segments
        transitions = []
        priorities = []
        indicies = []
        for ending in ending_points:
            transitions.append(self.buffer.get(ending)[2])
            priorities.append(torch.as_tensor(self.buffer.get(ending)[1]))
            indicies.append(torch.as_tensor(self.buffer.get(ending)[0]))
            
        return indicies, transitions, priorities
    
    def __len__(self):
        return len(self.buffer.nodes)
    
    def get_buffer(self):
        return repr(self.buffer)

prioritized_buffer = PriortizedBuffer(BUFFER_SIZE)

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return torch.argmax(policy_net(state)).view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.int)

episode_durations = []

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize(batch_size):
    if prioritized_buffer.counter < BUFFER_SIZE:
        return 
    # select transitions wrt the largest TD error calculated
    indicies, transitions, priorities = prioritized_buffer.sample(batch_size)


    batch = Transition(*zip(*transitions))

    # normalized the priorities so that the maximum is 1
    max_priority = max(priorities)
    normalized_priorities = [x / max_priority for x in priorities]

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])


    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action).to(torch.int64)
    reward_batch = torch.cat(batch.reward)
    priority_batch = torch.tensor(normalized_priorities, dtype=torch.float32, device=device)
    indicies_batch = torch.tensor(indicies, dtype=torch.float32, device=device)

    assert len(state_batch) == len(action_batch) == len(reward_batch), print(len(state_batch))

    action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)


    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    next_state_values = next_state_values.unsqueeze(1)

    expected_action_values = reward_batch + (GAMMA * next_state_values)

    # td-error for computing priority of samples
    td_errors = torch.abs(expected_action_values - action_values).cpu().detach().numpy()

    assert len(td_errors) == len(priority_batch) == len(indicies_batch), "sample index error"

    # now we update the ranking based on td-error
    prioritized_buffer.update_(indicies, td_errors)
    
    criterion = nn.SmoothL1Loss()
    
    loss = criterion(action_values, expected_action_values)
    optimizer_policy.zero_grad()
    loss.backward()
    optimizer_policy.step()


def main():
    # main method is used here
    # plot fucntion
    total_steps = 0
    for episode in range(N_EPS):
        if episode % 10 == 0:
            print("Eps: ", episode)
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            # take action based on epsilon-greedy
            action = select_action(state)

            # take a step with the action
            observation, reward, terminated, truncated, info = env.step(action.item())            
            
            # reward must be converted into tensor so that it can later on be concatenated in optimize_model
            reward = torch.tensor([[reward]], dtype=torch.float32, device=device)

            # check termination condition
            if terminated:
                next_state = None # this is used in optimize_model later
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            
            # add sample into buffer
            observation = torch.as_tensor(observation)
            assert torch.is_tensor(observation), "observation type error"

            prioritized_buffer.push(state, action, reward, next_state)

            # continue to the next state
            state = next_state
            optimize(BATCH_SIZE)

            prioritized_buffer.counter += 1
        
            # Soft update of the target network's weights
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if terminated or truncated:
                episode_durations.append(t+1)
                plot_durations()
                break

if __name__ == "__main__":
    main()
    print("Completed")
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()
    plt.close()

from collections import OrderedDict
import numpy as np
import copy
import random
import matplotlib.pyplot as plt

# import cv2
from typing import Dict, Tuple, List

from agent import DrunkenAgent, NaiveAgent
from simulator import Environment

############################################
############################################
class ReplayBuffer:
    def __init__(self):
        self.buffer = {"state":[], "action":[], "next_state":[], "reward":[], "done":[]}

    def add(self, state, action, next_state, reward, done):
        self.buffer["state"].append(state)
        self.buffer["action"].append(action)
        self.buffer["next_state"].append(next_state)
        self.buffer["reward"].append(reward)
        self.buffer["done"].append(done)

    def get_trajectory(self):
        return self.buffer

    def clear(self):
        self.buffer = []


def create_random_coordinate(x_bounds, y_bounds):
    x = random.uniform(*x_bounds)
    y = random.uniform(*y_bounds)
    return x, y


def create_random_start():
    chance = random.uniform(0, 1)
    if chance <= 0.33: # upper left
        x = random.uniform(2.5, 10)
        y = random.uniform(10, 17.5)
    elif chance <= 0.66: # lower left
        x = random.uniform(2.5, 10)
        y = random.uniform(2.5, 10)
    else: # lower right
        x = random.uniform(10, 17.5)
        y = random.uniform(2.5, 10)
    
    return x, y



def generate_random_trajectories(start_sample_area_interval, target_sample_area_interval, flow_field, num_rollouts, max_steps,
                                 num_actions, magnitude, threshold, action_type, penalty, render=False,
                                 fixed_target=False, target=None):
    ''' Fill up the replay buffer with random trajectories. '''
    buffer = ReplayBuffer()
    for i in range(num_rollouts):
        print("Rollout:", i)
        start = create_random_coordinate(*start_sample_area_interval)
        if not fixed_target:
            target = create_random_coordinate(*target_sample_area_interval)
        print("Start:", start)
        print("Target:", target)
        # agent = RandomAgent(momentum_length=3, action_space=NUM_ACTIONS, action_type="discrete")#action_space=((0, 1), (0, 1)))
        agent = NaiveAgent(target, num_actions=num_actions, magnitude=magnitude)
        env = Environment(flow_field, list(start), target, threshold=threshold,
                          action_type=action_type, num_actions=num_actions, magnitude=magnitude,
                          penalty=penalty)
        state = env.reset()
        done = False
        step = 0
        while not done and step < max_steps:
            action = agent.select_action(state)
            new_state, reward, done = env.step(action)
            buffer.add(state, action, new_state, reward, done)
            state = new_state
            #print(action) - works
            #print(new_state[0]) - works
            step += 1
        print("Last coordinate:", env.current_state[0])
        if render:
            env.render()
    return buffer


def generate_random_trajectories(start_sample_area_interval, target_sample_area_interval, flow_field, num_rollouts, max_steps,
                                 num_actions, magnitude, threshold, action_type, penalty, render=False):
    ''' Fill up the replay buffer with random trajectories. '''
    buffer = ReplayBuffer()
    for i in range(num_rollouts):
        print("Rollout:", i)
        start = create_random_coordinate(*start_sample_area_interval)
        target = create_random_coordinate(*target_sample_area_interval)
        print("Start:", start)
        print("Target:", target)
        # agent = RandomAgent(momentum_length=3, action_space=NUM_ACTIONS, action_type="discrete")#action_space=((0, 1), (0, 1)))
        agent = NaiveAgent(target, num_actions=num_actions, magnitude=magnitude)
        env = Environment(flow_field, list(start), target, threshold=threshold,
                          action_type=action_type, num_actions=num_actions, magnitude=magnitude,
                          penalty=penalty)
        state = env.reset()
        done = False
        step = 0
        while not done and step < max_steps:
            action = agent.select_action(state)
            new_state, reward, done = env.step(action)
            buffer.add(state, action, new_state, reward, done)
            state = new_state
            #print(action) - works
            #print(new_state[0]) - works
            step += 1
        print("Last coordinate:", env.current_state[0])
        if render:
            env.render()
    return buffer

def generate_random_trajectories_fixed_target(flow_field, num_rollouts, max_steps,
                                 num_actions, magnitude, threshold, action_type, penalty, render=False,
                                 target=None):
    ''' Fill up the replay buffer with random trajectories. '''
    buffer = ReplayBuffer()
    for i in range(num_rollouts):
        print("Rollout:", i)
        # start = create_random_start()
        start = create_random_coordinate((0, 20), (0, 20))
        random_target = create_random_coordinate((0, 20), (0, 20))
        print("Start:", start)
        print("Target:", target)
        print("Random target:", random_target)
        # agent = RandomAgent(momentum_length=3, action_space=NUM_ACTIONS, action_type="discrete")#action_space=((0, 1), (0, 1)))
        #agent = NaiveAgent(random_target, num_actions=num_actions, magnitude=magnitude)
        agent = DrunkenAgent(momentum_length=3 , target=random_target, num_actions=num_actions, magnitude=magnitude)
        env = Environment(flow_field, list(start), target, threshold=threshold,
                          action_type=action_type, num_actions=num_actions, magnitude=magnitude,
                          penalty=penalty)
        state = env.reset()
        done = False
        step = 0
        while not done and step < max_steps:
            action = agent.select_action(state)
            new_state, reward, done = env.step(action)
            # print("State:", state)
            # print("Action:", action)
            # print("New state:", new_state)
            # print("Reward:", reward)
            buffer.add(state, action, new_state, reward, done)
            # print("Buffer state", buffer.buffer["state"][-1])
            # print("Buffer action", buffer.buffer["action"][-1])
            # print("Buffer next state", buffer.buffer["next_state"][-1])
            # print("Buffer reward", buffer.buffer["reward"][-1])

            state = new_state.copy()
            #print(action) - works
            #print(new_state[0]) - works
            step += 1
        print("Last coordinate:", env.current_state[0])
        if i <= 5:
            fig, ax = env.render()
            fig.savefig(f"trajectory_plots/exploration/exploration_all_target_traj{i}.png")
            plt.close(fig)
    return buffer
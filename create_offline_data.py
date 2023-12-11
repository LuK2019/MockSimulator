import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import time 
import pickle
from agent import NaiveAgent

from simulator import Environment, SingleGyreFlowField

ACTION_TYPE = "discrete"
NUM_ACTIONS = 8
MAGNITUDE = 1 # Magnitude of the action vector
THRESHOLD = 1.0
PENALTY = True
RENDER = False

NUM_ROLLOUTS = 100
MAX_STEPS = 100


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

def generate_random_trajectories(start_sample_area_interval, target_sample_area_interval, flow_field, num_rollouts, max_steps):
    ''' Fill up the replay buffer with random trajectories. '''
    buffer = ReplayBuffer()
    for i in range(num_rollouts):
        print("Rollout:", i)
        start = create_random_coordinate(*start_sample_area_interval)
        target = create_random_coordinate(*target_sample_area_interval)
        print("Start:", start)
        print("Target:", target)
        # agent = RandomAgent(momentum_length=3, action_space=NUM_ACTIONS, action_type="discrete")#action_space=((0, 1), (0, 1)))
        agent = NaiveAgent(target, NUM_ACTIONS, magnitude=MAGNITUDE)
        env = Environment(flow_field, list(start), target, threshold=THRESHOLD,
                          action_type=ACTION_TYPE, num_actions=NUM_ACTIONS, magnitude=MAGNITUDE,
                          penalty=PENALTY)
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
        if RENDER:
            env.render()
    return buffer




if __name__ == "__main__":
    # Usage of random trajectories
    # action_space = ((0, 1), (0, 1)) 
    flow_field = SingleGyreFlowField(width=20, height=20, center=(10, 10), radius=4, strength=1)
    buffer = generate_random_trajectories(
        start_sample_area_interval=[(1,3),(1,3)],
        target_sample_area_interval=[(16, 19), (16, 19)],
        flow_field=flow_field,
        num_rollouts=NUM_ROLLOUTS,
        max_steps=MAX_STEPS)
    print("Number of trajectories:", len(buffer.buffer))
    # Pickl the buffer and dump it to a file 
    current_time = time.strftime("%Y%m%d-%H%M%S")
    filename = f"logs/buffer_{current_time}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(buffer.buffer, f)
        print("Buffer saved to", filename)
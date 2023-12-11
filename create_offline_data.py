import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import time 
import pickle
from agent import NaiveAgent

from simulator import Environment, SingleGyreFlowField
from utils_custom import generate_random_trajectories, generate_random_trajectories_fixed_target

ACTION_TYPE = "discrete"
NUM_ACTIONS = 8
MAGNITUDE = 1 # Magnitude of the action vector
THRESHOLD = 1.0
PENALTY = True
RENDER = False

NUM_ROLLOUTS = 10000
MAX_STEPS = 30



if __name__ == "__main__":
    # Usage of random trajectories
    # action_space = ((0, 1), (0, 1)) 
    flow_field = SingleGyreFlowField(width=20, height=20, center=(10, 10), radius=4, strength=1)
    # buffer = generate_random_trajectories(
    #     start_sample_area_interval=[(1,3),(1,3)],
    #     target_sample_area_interval=[(16, 19), (16, 19)],
    #     flow_field=flow_field,
    #     num_rollouts=NUM_ROLLOUTS,
    #     max_steps=MAX_STEPS,
    #     num_actions=NUM_ACTIONS,
    #     magnitude=MAGNITUDE,
    #     threshold=THRESHOLD,
    #     action_type=ACTION_TYPE,
    #     penalty=PENALTY)
    buffer = generate_random_trajectories_fixed_target(
        flow_field=flow_field,
        num_rollouts=NUM_ROLLOUTS,
        max_steps=MAX_STEPS,
        num_actions=NUM_ACTIONS,
        magnitude=MAGNITUDE,
        threshold=THRESHOLD,
        action_type=ACTION_TYPE,
        penalty=PENALTY,
        target = [17.5, 17.5])


    print("Number of transitions:", len(buffer.buffer["done"]))
    # plot buffer.buffer["state"] as a scatter plot in a 2D plane of 20 x 20
    fig, ax = plt.subplots()
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_aspect('equal')
    ax.grid(True, which='both')
    ax.set_xticks(np.arange(0, 21, 1))
    ax.set_yticks(np.arange(0, 21, 1))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("State_distribution")
    x_vals = [x[0] for x in buffer.buffer["state"]]
    y_vals = [x[1] for x in buffer.buffer["state"]]
    ax.scatter(x_vals, y_vals, s=1, c="blue", alpha=0.3)
    plt.show()

    # Pickl the buffer and dump it to a file 
    current_time = time.strftime("%Y%m%d-%H%M%S")
    filename = f"logs/buffer_{current_time}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(buffer.buffer, f)
        print("Buffer saved to", filename)
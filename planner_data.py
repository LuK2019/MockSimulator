import pickle
import numpy as np
import matplotlib.pyplot as plt
import os 
import h5py


# Load the data
log_name = 'buffer_20231213-055554'

def reformat_obs(obs):
    pos = np.array(obs[0], dtype=np.float32)
    vectorfield = obs[1].flatten() # Flatten array of shape (20, 20, 2, 2)
    # Change dtype to float32
    vectorfield = np.array(vectorfield, dtype=np.float32)
    pos_and_vectorfield = np.concatenate((pos, vectorfield))
    return pos_and_vectorfield

if __name__ == "__main__":
    with open(os.path.join('logs', log_name+'.pkl'), 'rb') as f:
        buffer = pickle.load(f)

    all_actions = [experience["action"] for experience in buffer]
    all_positions = [experience["state"][0] for experience in buffer]
    all_rewards = [experience["reward"] for experience in buffer]
    all_terminations = [experience["done"] for experience in buffer]

    j = 0
    cut_off = 40
    trajectories = []

    for i in range(len(all_actions)):
        if all_terminations[i]:
            trajectory = [(all_positions[k], all_actions[k]) for k in range(j, i + 1)]
            total_reward = sum(all_rewards[j:i+1])

            if len(trajectory) > cut_off:
                trajectory = trajectory[:cut_off]
                trajectories.append((trajectory, total_reward))
            j = i + 1

    with open(os.path.join('logs', 'planner_' + log_name + '.pkl'), 'wb') as f:
        pickle.dump(trajectories, f)
        print(f"Stored trajectories in {'logs/' + 'planner_' + log_name + '.pkl'}")




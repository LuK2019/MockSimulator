import pickle
import numpy as np
import matplotlib.pyplot as plt
import os 
import h5py

# Load the data
log_name = 'buffer_20231202-115430'

def reformat_obs(obs):
    pos = np.array(obs[0], dtype=np.float32)
    vectorfield = obs[1].flatten() # Flatten array of shape (20, 20, 2, 2)
    # Change dtype to float32
    vectorfield = np.array(vectorfield, dtype=np.float32)
    pos_and_vectorfield = np.concatenate((pos, vectorfield))
    return pos_and_vectorfield

with open(os.path.join('logs', log_name+'.pkl'), 'rb') as f:
    buffer = pickle.load(f)
print(buffer[0].keys())
# Plot the positions of the agent on a grid
data = [experience["state"][0] for experience in buffer]
data = np.array(data)
print(data.shape)
plt.scatter(data[:,0], data[:,1])
# plt.show()

flattened_buffer = {}
actions = []
observations = []
rewards = []
terminals = []
timeouts = []
for transition in buffer:
    actions.append(transition["action"])
    obs = transition["state"]
    obs = reformat_obs(obs)
    observations.append(obs)
    rewards.append(transition["reward"])
    terminals.append(transition["done"])
    timeouts.append(False)

flattened_buffer["actions"] = np.array(actions)
flattened_buffer["observations"] = np.array(observations)
flattened_buffer["rewards"] = np.array(rewards)
flattened_buffer["terminals"] = np.array(terminals)
flattened_buffer["timeouts"] = np.array(timeouts)


with h5py.File('logs/'+'flattened_'+log_name + '.h5', 'w') as f:
    for key, value in flattened_buffer.items():
        f.create_dataset(key, data=value)

print(f"Stored data in {'logs/'+'flattened_'+log_name + '.h5'}")

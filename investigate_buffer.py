import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the data
with open('buffer_20231202-115430.pkl', 'rb') as f:
    buffer = pickle.load(f)
print(buffer[0].keys())
# Plot the positions of the agent on a grid
data = [experience["state"][0] for experience in buffer]
data = np.array(data)
print(data.shape)
plt.scatter(data[:,0], data[:,1])
plt.show()
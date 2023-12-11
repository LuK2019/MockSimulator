import pickle
import numpy as np
import os

log_name = "buffer_20231210-182008"
with open(os.path.join('logs', log_name+'.pkl'), 'rb') as f:
    buffer = pickle.load(f)


class ReplayBuffer_from_file:
    def __init__(self, data_dict):
        
        self.observations = np.array(data_dict["state"])
        self.actions = np.array(data_dict["action"])
        self.next_observations = np.array(data_dict["next_state"])
        self.rewards = np.array(data_dict["reward"])
        self.dones = np.array(data_dict["done"])
        self.size = len(self.dones)

        assert len(self.observations) == len(self.next_observations)


    def sample(self, batch_size):
        rand_indices = np.random.randint(0, self.size, size=(batch_size,)) % self.size
        return {
            "observations": self.observations[rand_indices],
            "actions": self.actions[rand_indices],
            "rewards": self.rewards[rand_indices],
            "next_observations": self.next_observations[rand_indices],
            "dones": self.dones[rand_indices],
        }

    def __len__(self):
        return self.size


#dataset = ReplayBuffer_flattened(retrieved_dict)
dataset = ReplayBuffer_from_file(buffer)

print(dataset.observations.shape)
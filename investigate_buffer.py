import pickle
import numpy as np
import matplotlib.pyplot as plt
import os 
import h5py

MODE = "LUKAS_DIFFUSOR_TEST_1_FORMAT"
#MODE = "BEAR_DIFFUSOR_FORMAT"
# Load the data
log_name = 'buffer_20231207-190028'

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

    if MODE == "LUKAS_DIFFUSOR_TEST_1_FORMAT":
        all_actions = [experience["action"] for experience in buffer]
        all_positions = [experience["state"][0] for experience in buffer]
        all_terminations = [experience["done"] for experience in buffer]

        # Filter all actions where the interval length is less than 100 until termination
        all_actions = np.array(all_actions)
        all_positions = np.array(all_positions)
        all_terminations = np.array(all_terminations)
        j = 0
        all_actions_interval = []
        all_positions_interval = []
        for i in range(0, len(all_actions)):
            if all_terminations[i] == True:
                actions_interval = all_actions[j:i+1]
                all_actions_interval.append(actions_interval)
                positions_interval = all_positions[j:i+1]
                all_positions_interval.append(positions_interval)
                print("j:i ", j,":", i)
                print("all_positions_interval: ", len(all_positions_interval))
                j = i+1
        
        # reshape all_actions to a matrix with 100 columns
        cut_off = 40
        all_actions = [sequence[:cut_off] for sequence in all_actions_interval if len(sequence) >= cut_off]
        all_positions = [sequence[:cut_off] for sequence in all_positions_interval if len(sequence) >= cut_off]
        all_actions = np.array(all_actions).reshape(-1,cut_off, 2)
        all_positions = np.array(all_positions).reshape(-1,cut_off,2)
        print("all_actions: ", all_actions.shape)
        print("all_positions: ", all_positions.shape)

        all_actions = np.swapaxes(all_actions, 1, 2)
        print("all_actions: ", all_actions.shape)
        all_positions = np.swapaxes(all_positions, 1, 2)
        print("all_positions: ", all_positions.shape)

        # Plot the positions of the agent on a grid and add the corresponding action as a vector on that point
        for i in range(0, all_actions.shape[0], 100):
            # fix the x and y axis extend to [0, 20]
            plt.xlim(0, 20)
            plt.ylim(0, 20)
            plt.scatter(all_positions[i:i+100,0,:], all_positions[i:i+100,1,:])
            # plt.show()

        # all_positions and all_actions have shape (number of trajectories, action_dim, cut_off)
        # now i want a vector of shape (number of trajectories, action_dim+state_dim, cut_off)
        # where the first two entries are action and the rest is state
        concatenated_data = np.concatenate((all_actions, all_positions), axis=1)
        print("concatenated_data: ", concatenated_data.shape)

        # Now save this array as pickel object
        with open(os.path.join('logs', 'concatenated_'+log_name+'.pkl'), 'wb') as f:
            pickle.dump(concatenated_data, f)
            print(f"Stored data in {'logs/'+'concatenated_'+log_name+'.pkl'}")
         

    elif MODE == "BEAR_DIFFUSOR_FORMAT":
        # Plot the positions of the agent on a grid
        data = [experience["state"][0] for experience in buffer]
        data = np.array(data)
        print(data.shape)
        plt.scatter(data[:,0], data[:,1])
        plt.show()

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
    else:
        pass

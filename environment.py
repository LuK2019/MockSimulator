import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

NUM_ACTIONS = 8
MAGNITUDE = 1 # Magnitude of the action vector
RENDER = False


class FlowField:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def get_flow_at_position(self, x, y):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_flow_grid(self, resolution=20):
        ''' Returns a grid of shape (width, height, 2, 2) where the penultimate dimension is the x,y coordinate and the last dimension represents the flow vector at each position.
        The resolution parameter determines how many points are sampled in each direction.'''

        grid_x = np.linspace(0, self.width, resolution)
        grid_y = np.linspace(0, self.height, resolution)
        flow_grid = np.zeros((resolution, resolution, 2, 2))

        for i, x in enumerate(grid_x):
            for j, y in enumerate(grid_y):
                flow_grid[i, j, :, 0] = [x, y]
                flow_grid[i, j, :, 1] = self.get_flow_at_position(x, y)

        return flow_grid


class UniformFlowField(FlowField):
    def __init__(self, height, width, flow_vector):
        super().__init__(height, width)
        self.flow_vector = flow_vector
    
    def get_flow_at_position(self, x, y):
        if 0 <= x <= self.height and 0 <= y <= self.width:
            return self.flow_vector
        else:
            raise ValueError(f"Position ({x}, {y}) is outside the flow field.")

class SegmentedFlowField(UniformFlowField):
    def __init__(self, height, width, flow_vector):
        super().__init__(height, width, flow_vector)

    def get_flow_at_position(self, x, y):
        if 0 <= x <= self.height and 0 <= y <= self.width:
            third_height = self.height // 3
            if third_height <= x < 2 * third_height:
                return self.flow_vector
            else:
                return (0, 0)
        else:
            raise ValueError(f"Position ({x}, {y}) is outside the flow field.")

class SingleGyreFlowField(FlowField):
    def __init__(self, height, width, center, radius, strength):
        super().__init__(height, width)
        self.center = center
        self.radius = radius
        self.strength = strength

    def get_flow_at_position(self, y, x):  # Swap x and y in the parameter list
        if 0 <= y <= self.height and 0 <= x <= self.width:
            dx = x - self.center[1]  # Swap center coordinates
            dy = y - self.center[0]
            distance = np.sqrt(dx**2 + dy**2)
            if distance < self.radius:
                return (-self.strength * dy, self.strength * dx)
            else:
                return (0, 0)
        else:
            raise ValueError(f"Position ({x}, {y}) is outside the flow field.")  # Swap x and y in the error message



class Agent:
    def __init__(self, loaded_policy=None, action_type="continous"):
        self.loaded_policy = loaded_policy
        self.action_type = action_type

    def select_action(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

class RandomAgent(Agent):
    def __init__(self, action_space, action_type="continous", momentum_length=None):
        super().__init__(action_type=action_type)
        self.action_space = action_space # boundaries of the action space if continous; num of actions in discrete

        self.momentum_length = momentum_length
        self.actions_in_a_row = 0
        self.last_action = None

    def select_action(self, observation):
        if self.action_type == "continous":
            x_action = np.random.uniform(self.action_space[0][0], self.action_space[0][1])
            y_action = np.random.uniform(self.action_space[1][0], self.action_space[1][1])
            return (x_action, y_action)
        else:
            if self.actions_in_a_row < self.momentum_length and self.last_action is not None:
                self.actions_in_a_row += 1
                return self.last_action
            else:
                self.actions_in_a_row = 0
                self.last_action = np.random.choice(self.action_space)
                return self.last_action
            
        
    
class UniformAgent(Agent):
    def __init__(self, uniform_action,action_type="continous"):
        super().__init__(action_type)
        self.uniform_action = uniform_action # number in discrete; tuple is continous

    def select_action(self, observation):
        return self.uniform_action
    

class NaiveAgent(Agent):
    def __init__(self, target, num_actions, action_type="discrete"):
        super().__init__(action_type=action_type)
        self.target = target
        self.num_actions = num_actions
        self.action_map = {i: (np.cos(2 * np.pi * i / num_actions), np.sin(2 * np.pi * i / num_actions)) for i in range(num_actions)}

    def select_action(self, observation):
        # Calculate the angle between the current position and the target
        dx = self.target[0] - observation[0]
        dy = self.target[1] - observation[0]
        angle = np.arctan2(dy, dx)

        if self.action_type == "continous":
            return (np.cos(angle), np.sin(angle)) # TODO: Adjust Magnitude of Acttion here
        else:
            # Calculate the index of the action that is closest to the angle
            action = int(np.round(angle * self.num_actions / (2 * np.pi))) % self.num_actions

        return action

class Environment:
    def __init__(self, flow_field, initial_xy, target, threshold, buffer,
                 action_type="continous", num_actions=None, magnitude = MAGNITUDE): # initial_xy - list
        self.flow_field = flow_field
        self.target = target
        self.threshold = threshold
        self.replay_buffer = buffer
        self.initial_state = initial_xy.copy()
        self.current_state = initial_xy.copy()
        self.history = [self.initial_state.copy()]
        self.action_type = action_type
        self.num_actions = num_actions
        self.magnitude = magnitude

        if self.action_type == "discrete":
            # Define the number of actions
            assert num_actions is not None, "Please specify the number of actions."

            # Calculate the angle of each action
            angles = np.linspace(0, 2 * np.pi, num_actions, endpoint=False)

            # Set the length of each vector
            length = magnitude

            # Calculate the x, y components of each vector
            x = length * np.cos(angles)
            y = length * np.sin(angles)

            self.action_map = {i: (x[i], y[i]) for i in range(num_actions)}
    
    def get_initial_state(self):
        return self.initial_state

    def step(self, action):
        state = self.current_state
        #print("Current state:", state)
        flow = self.flow_field.get_flow_at_position(*self.current_state) # based on current position
        if self.action_type == "continous":
            self.current_state[0] += action[0] + flow[0] # update x
            self.current_state[1] += action[1] + flow[1] # update y
        elif self.action_type == "discrete":
            self.current_state[0] += self.action_map[action][0] + flow[0]
            self.current_state[1] += self.action_map[action][1] + flow[1]
        #print("New state:", self.current_state)
        self.history.append(self.current_state.copy())
        #print("History:", self.history)
        new_state = self.current_state
        reward = self.compute_reward(new_state)
        done = self.is_done()

        self.replay_buffer.add(state, action, new_state, reward, done)
        return new_state, action, reward, done

    def reset(self):
        current_state = self.initial_state.copy()  # Reset agent to initial state
        self.history = [self.initial_state.copy()]
        # reward = self.compute_reward(self.initial_state[0])
        return self.current_state

    def compute_reward(self, position):
        # Calculate the Euclidean distance from the current position to the target
        distance = ((position[0] - self.target[0]) ** 2 + (position[1] - self.target[1]) ** 2) ** 0.5
        # Negative of the distance as the reward
        return -distance

    def is_done(self):
        # Check if the agent is outside the grid
        if not (0 <= self.current_state[0] < self.flow_field.width and 0 <= self.current_state[1] < self.flow_field.height):
            outside_of_grid = True
            print("Episode terminated: Agent moved outside the grid.")
            return outside_of_grid
        
        # Calculate the distance between the agent and the target
        distance = ((self.current_state[0] - self.target[0]) ** 2 + (self.current_state[1] - self.target[1]) ** 2) ** 0.5
        reached_goal = (distance <= self.threshold)
        return reached_goal

    def render(self):
        X, Y = np.mgrid[0:self.flow_field.width, 0:self.flow_field.height]
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        for i in range(self.flow_field.height):  # Swap the loops
            for j in range(self.flow_field.width):
                U[i, j], V[i, j] = self.flow_field.get_flow_at_position(j, i)  # Swap i and j here

        plt.quiver(X, Y, U, V, pivot='mid')

        # Plot the agent's trajectory
        #plot(self.history)
        x_vals, y_vals = zip(*self.history)
        plt.plot(x_vals, y_vals, 'ro-')  # 'ro-' means red color, circle markers, and solid line

        # Mark the agent's current position
        plt.plot(self.current_state[0], self.current_state[1], 'bo')  # 'bo' means blue color and circle markers

        # Draw the target as a green box
        target_rect = patches.Rectangle(self.target, 1, 1, linewidth=1, edgecolor='g', facecolor='none')
        plt.gca().add_patch(target_rect)

        # Set plot limits and labels
        plt.xlim(0, self.flow_field.width)
        plt.ylim(0, self.flow_field.height)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Agent Movement in Flow Field')

        # Show the plot
        plt.show()
        
# class ReplayBuffer:
#     def __init__(self):
#         self.buffer = []

#     def add(self, state, action, next_state, reward, done):
#         experience = {"state": state, "action":action, "next_state":next_state, "reward":reward, "done": done}
#         self.buffer.append(experience)

#     def get_trajectory(self):
#         return self.buffer

#     def clear(self):
#         self.buffer = []

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



import random

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
        agent = NaiveAgent(target, NUM_ACTIONS)
        env = Environment(flow_field, list(start), target, threshold=1.0,
                           buffer=buffer, action_type="discrete", num_actions=NUM_ACTIONS)
        state = env.reset()
        done = False
        step = 0
        while not done and step < max_steps:
            action = agent.select_action(state)
            new_state, action, reward, done = env.step(action)
            #print(action) - works
            #print(new_state[0]) - works
            step += 1
        print("Last coordinate:", env.current_state[0])
        if RENDER:
            env.render()
    return buffer


import time 
import pickle

if __name__ == "__main__":
    variant = 2

    if variant == 1:
        # Example usage with custom action space
        action_space = ((0, 1), (0, 1))  # Agents can move between -2 and 2 steps in both x and y
        flow_field = SingleGyreFlowField(width=10, height=10, center=(5, 5), radius=3, strength=1)
        #flow_field = UniformFlowField(width=10, height=10, flow_vector=(1, 0), action_space=action_space)
        #flow_field = SegmentedFlowField(width=10, height=10, flow_vector=(1, 0))
        print("Custom Action Space:", flow_field.action_space)
        uniform_agent = UniformAgent(uniform_action= (1.0, 0.5))  # UniformAgent always moves (0.5, 0.5)
        random_agent = RandomAgent([(1, 0), (0, 1), (-1, 0), (0, -1)])  # RandomAgent chooses randomly 
        NUM_ROLLOUTS = 10
        env = Environment(flow_field, [0, 0], target=(8, 8), threshold=1.0)
        render = True
        state = env.reset()
        done = False
        print("Initial State:", state)  
        for i in range(NUM_ROLLOUTS):
            print("Rollout:", i)
            done = False
            while not done:
                new_state, action, reward, done = env.step()
                print("State:", new_state, "Action:", action, "Reward:", reward)
                if render:
                    env.render()
            env.reset()
    else:
        # Usage of random trajectories
        action_space = ((0, 1), (0, 1)) 
        flow_field = SingleGyreFlowField(width=20, height=20, center=(10, 10), radius=4, strength=1)
        buffer = generate_random_trajectories(
            start_sample_area_interval=[(1,3),(1,3)],
            target_sample_area_interval=[(16, 19), (16, 19)],
            flow_field=flow_field,
            num_rollouts=100,
            max_steps=100)
        print("Number of trajectories:", len(buffer.buffer))
        # Pickl the buffer and dump it to a file 
        current_time = time.strftime("%Y%m%d-%H%M%S")
        filename = f"logs/buffer_{current_time}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(buffer.buffer, f)
            print("Buffer saved to", filename)
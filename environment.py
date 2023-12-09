import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class FlowField:
    def __init__(self, height, width, action_space=None):
        self.height = height
        self.width = width
        if action_space is None:
            self.action_space = ((-1, 1), (-1, 1))
        else:
            self.action_space = action_space

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
    def __init__(self, height, width, flow_vector, action_space=None):
        super().__init__(height, width, action_space)
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
    def __init__(self, height, width, center, radius, strength, action_space=None):
        super().__init__(height, width, action_space)
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
    def __init__(self, action_space=None, loaded_policy=None):
        self.action_space = action_space
        self.loaded_policy = loaded_policy

    def select_action(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

class RandomAgent(Agent):
    def __init__(self, action_space):
        super().__init__(action_space)

    def select_action(self, observation):
        x_action = np.random.uniform(self.action_space[0][0], self.action_space[0][1])
        y_action = np.random.uniform(self.action_space[1][0], self.action_space[1][1])
        return (x_action, y_action)
    
class UniformAgent(Agent):
    def __init__(self, uniform_action):
        super().__init__()
        self.uniform_action = uniform_action

    def select_action(self, observation):
        return self.uniform_action

class Environment:
    def __init__(self, flow_field, initial_xy, target, threshold, buffer): # initial_xy - list
        self.flow_field = flow_field
        self.target = target
        self.threshold = threshold
        self.replay_buffer = buffer
        self.initial_state = (initial_xy.copy(), self.flow_field.get_flow_grid())
        self.current_state = (initial_xy.copy(), self.flow_field.get_flow_grid())
        self.history = [self.initial_state[0]]
    
    def get_initial_state(self):
        return self.initial_state

    def step(self, action):
        state = self.current_state
        flow = self.flow_field.get_flow_at_position(*self.current_state[0]) # based on current position
        self.current_state[0][0] += action[0] + flow[0] # update x
        self.current_state[0][1] += action[1] + flow[1] # update y
        self.history.append(self.current_state[0])

        new_state = self.current_state
        reward = self.compute_reward(new_state[0])
        done = self.is_done()

        self.replay_buffer.add(state, action, new_state, reward, done)
        return new_state, action, reward, done

    def reset(self):
        self.current_state[0][0], self.current_state[0][1] = self.initial_state[0]  # Reset agent to initial state
        self.history = [self.initial_state[0]]
        # reward = self.compute_reward(self.initial_state[0])
        return (self.current_state[0], *self.flow_field.get_flow_grid())

    def compute_reward(self, position):
        # Calculate the Euclidean distance from the current position to the target
        distance = ((position[0] - self.target[0]) ** 2 + (position[1] - self.target[1]) ** 2) ** 0.5
        # Negative of the distance as the reward
        return -distance

    def is_done(self):
        # Check if the agent is outside the grid
        if not (0 <= self.current_state[0][0] < self.flow_field.width and 0 <= self.current_state[0][1] < self.flow_field.height):
            outside_of_grid = True
            print("Episode terminated: Agent moved outside the grid.")
            return outside_of_grid
        
        # Calculate the distance between the agent and the target
        distance = ((self.current_state[0][0] - self.target[0]) ** 2 + (self.current_state[0][1] - self.target[1]) ** 2) ** 0.5
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
        plt.plot(self.current_state[0][0], self.current_state[0][1], 'bo')  # 'bo' means blue color and circle markers

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
        
class ReplayBuffer:
    def __init__(self):
        self.buffer = []

    def add(self, state, action, next_state, reward, done):
        experience = {"state": state, "action":action, "next_state":next_state, "reward":reward, "done": done}
        self.buffer.append(experience)

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
        agent = RandomAgent(action_space=action_space)
        env = Environment(flow_field, list(start), target, threshold=1.0, buffer=buffer)
        state = env.reset()
        done = False
        step = 0
        while not done and step < max_steps:
            action = agent.select_action(state)
            new_state, action, reward, done = env.step(action)
            step += 1
        print("Last coordinate:", env.current_state[0])
        env.render()
    return buffer


import time 
import pickle

if __name__ == "__main__":
    variant = 2

    if variant == 1:
        # Example usage with custom action space
        action_space = ((0, 1), (0, 1))  # Agents can move between -2 and 2 steps in both x and y
        flow_field = SingleGyreFlowField(width=10, height=10, center=(5, 5), radius=3, strength=1, action_space=action_space)
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
        flow_field = SingleGyreFlowField(width=20, height=20, center=(10, 10), radius=4, strength=1, action_space=action_space)
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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class FlowField:
    def __init__(self, height, width, action_space=None):
        self.height = height
        self.width = width
        if action_space is None:
            # Default action space: interval for x and y (e.g., -1 to 1 for both x and y)
            self.action_space = ((-1, 1), (-1, 1))
        else:
            self.action_space = action_space

    def get_vector_field(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_flow_at_position(self, x, y):
        if self.field is None:
            raise ValueError("get_flow_at_poistion: Flow field is not initialized.")
        if 0 <= x < self.height and 0 <= y < self.width:
            return self.field[x,y,...]
        else:
            raise ValueError("get_flow_at_poistion: Position ({}, {}) is outside the flow field.".format(x, y))

class UniformFlowField(FlowField):
    def __init__(self, height, width, flow_vector, action_space=None):
        super().__init__(height, width, action_space)
        self.flow_vector = flow_vector
        self.field = self.get_vector_field()
    
    def get_vector_field(self):
        field = np.empty((self.height, self.width, 2))
        field[:] = self.flow_vector
        return field

class SegmentedFlowField(UniformFlowField):
    def __init__(self, height, width, flow_vector):
        super().__init__(height, width, flow_vector)
        self.field = self.get_vector_field()

    def get_vector_field(self):
        field = np.zeros((self.height, self.width, 2))  # Initialize with zero flow
        third_height = self.height // 3
        for x in range(third_height, 2 * third_height):
            for y in range(self.width):
                field[x, y,...] = self.flow_vector  # Set flow vector in the middle third
        return field
        
class SingleGyreFlowField(FlowField):
    def __init__(self, height, width, center, radius, strength, action_space=None):
        super().__init__(height, width, action_space)
        self.center = center  # Center of the gyre (x, y)
        self.radius = radius  # Radius of the gyre
        self.strength = strength  # Strength of the gyre
        self.field = self.get_vector_field()

    def get_vector_field(self):
        field = np.zeros((self.height, self.width, 2))  # 2 for x and y components
        for x in range(self.height):
            for y in range(self.width):
                # Calculate the distance from the center
                dx = x - self.center[0]
                dy = y - self.center[1]
                distance = np.sqrt(dx**2 + dy**2)

                # Check if the point is within the gyre's influence
                if distance < self.radius:
                    field[x, y, 0] = -self.strength * dy # x component
                    field[x, y, 1] = self.strength * dx   # y component

        assert field.shape == (self.height, self.width, 2), "Flow field has incorrect shape."
        return field

class Agent:
    def __init__(self, start_x, start_y, action_space=None):
        self.x = start_x
        self.y = start_y
        self.action_space = action_space
        self.history = [(start_x, start_y)]

    def move(self, action, flow_field):
        flow = flow_field.get_flow_at_position(self.x, self.y)
        self.x += action[0] + flow[0]
        self.y += action[1] + flow[1]
        self.history.append((self.x, self.y))

    def get_current_position(self):
        return self.x, self.y

    def select_action(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

class RandomAgent(Agent):
    def __init__(self, start_x, start_y, action_space):
        super().__init__(start_x, start_y, action_space)

    def select_action(self):
        x_action = np.random.uniform(self.action_space[0][0], self.action_space[0][1])
        y_action = np.random.uniform(self.action_space[1][0], self.action_space[1][1])
        return (x_action, y_action)
    
class UniformAgent(Agent):
    def __init__(self, start_x, start_y, uniform_action):
        super().__init__(start_x, start_y)
        self.uniform_action = uniform_action

    def select_action(self):
        return self.uniform_action

class Environment:
    def __init__(self, flow_field, agent, target, threshold):
        self.flow_field = flow_field
        self.agent = agent  # Use consistent naming for the agent
        self.target = target
        self.threshold = threshold
        self.replay_buffer = ReplayBuffer()
        self.initial_state = ((agent.x, agent.y), self.flow_field.field)  # Store the initial state for reset

    def step(self):
        action = self.agent.select_action()  # Use self.agent
        current_state = (self.agent.get_current_position(), self.flow_field.field)
        self.agent.move(action, self.flow_field)
        new_state = (self.agent.get_current_position(), self.flow_field.field)
        reward = self.compute_reward(new_state[0])
        done = self.is_done()
        # Check if the agent is outside the grid
        if not (0 <= self.agent.x < self.flow_field.width and 0 <= self.agent.y < self.flow_field.height):
            done = True
            print("Episode terminated: Agent moved outside the grid.")
        self.replay_buffer.add(current_state, action, new_state, reward)
        return new_state, action, reward, done

    def reset(self):
        self.agent.x, self.agent.y = self.initial_state[0]  # Reset agent to initial state
        self.agent.history = [self.initial_state[0]]
        reward = self.compute_reward(self.initial_state[0])
        return (self.agent.get_current_position(), self.flow_field.field)

    def compute_reward(self, position):
        # Calculate the Euclidean distance from the current position to the target
        distance = ((position[0] - self.target[0]) ** 2 + (position[1] - self.target[1]) ** 2) ** 0.5
        # Negative of the distance as the reward
        return -distance

    def is_done(self):
        # Calculate the distance between the agent and the target
        distance = ((self.agent.x - self.target[0]) ** 2 + (self.agent.y - self.target[1]) ** 2) ** 0.5
        return distance <= self.threshold

    def render(self):
        # Create a grid of arrows to represent the flow field
        X, Y = np.mgrid[0:self.flow_field.height, 0:self.flow_field.width]
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        for i in range(self.flow_field.width):
            for j in range(self.flow_field.height):
                U[i, j], V[i, j] = self.flow_field.get_flow_at_position(i, j)

        plt.quiver(X, Y, U, V, pivot='mid')

        # Plot the agent's trajectory
        x_vals, y_vals = zip(*self.agent.history)
        plt.plot(x_vals, y_vals, 'ro-')  # 'ro-' means red color, circle markers, and solid line

        # Mark the agent's current position
        plt.plot(self.agent.x, self.agent.y, 'bo')  # 'bo' means blue color and circle markers

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

    def add(self, state, action, next_state, reward):
        experience = (state, action, next_state, reward)
        self.buffer.append(experience)

    def get_trajectory(self):
        return self.buffer

    def clear(self):
        self.buffer = []


if __name__ == "__main__":

    # Example usage with custom action space
    action_space = ((0, 1), (0, 1))  # Agents can move between -2 and 2 steps in both x and y
    # flow_field = SingleGyreFlowField(width=10, height=10, center=(5, 5), radius=3, strength=2, action_space=action_space)
    flow_field = UniformFlowField(width=10, height=10, flow_vector=(0.5, 0.5), action_space=action_space)
    print("Custom Action Space:", flow_field.action_space)
    uniform_agent = UniformAgent(start_x=2, start_y=2, uniform_action= (0.5, 0.5))  # UniformAgent always moves (0.5, 0.5)
    random_agent = RandomAgent(0, 0, [(1, 0), (0, 1), (-1, 0), (0, -1)])  # RandomAgent chooses randomly

    env = Environment(flow_field, uniform_agent, target=(8, 8), threshold=1.0)
    render = True
    state = env.reset()
    done = False
    print("Initial State:", state)  
    while not done:
        new_state, action, reward, done = env.step()
        print("State:", new_state, "Action:", action, "Reward:", reward)
        if render:
            env.render()


import numpy as np
import random

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
    def __init__(self, target, num_actions, action_type="discrete", magnitude=1):
        super().__init__(action_type=action_type)
        self.target = target
        self.num_actions = num_actions
        self.magnitude = magnitude
        self.action_map = {i: (np.cos(2 * np.pi * i / num_actions), np.sin(2 * np.pi * i / num_actions)) for i in range(num_actions)}

    def select_action(self, observation):
        # Calculate the angle between the current position and the target
        dx = self.target[0] - observation[0]
        dy = self.target[1] - observation[1]
        angle = np.arctan2(dy, dx)

        if self.action_type == "continous":
            return (self.magnitude * np.cos(angle), self.magnitude * np.sin(angle))
        else:
            # Calculate the index of the action that is closest to the angle
            action = int(np.round(angle * self.num_actions / (2 * np.pi))) % self.num_actions
            #print("----------------")
            #print("Action: ", action)
            #print("Angle: ", (180 / np.pi) * angle)
            #print("----------------")

        return action
    
# make a new agent called DrunkenAgent which by chance acts either like the RandomAgent or the NaiveAgent

class DrunkenAgent(Agent):
    def __init__(self, action_space, target, num_actions, action_type="discrete", momentum_length=3, magnitude=1):
        super().__init__(action_type=action_type)
        # Initialize RandomAgent and NaiveAgent objects
        self.last_agent_type = ...
        self.random_agent = RandomAgent(action_space, action_type, momentum_length)
        self.naive_agent = NaiveAgent(target, num_actions, action_type, magnitude)

    def select_action(self, observation):
        # Randomly choose to use the action from RandomAgent or NaiveAgent
        if random.choice([True, False]):
            return self.random_agent.select_action(observation)
        else:
            return self.naive_agent.select_action(observation)


class CQLAgent(Agent):
    def __init__(self, loaded_policy=None, action_type="discrete"):
        assert loaded_policy is not None, "Please specify path to CQL"
        self.loaded_policy = loaded_policy
        self.action_type = action_type

    def select_action(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
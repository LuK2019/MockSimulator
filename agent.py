import numpy as np
import random
import torch.nn as nn
import torch
import pytorch_util as ptu

from utils_diffusor import load_checkpoint, get_model, reset_start_and_target, limits_normalizer, limits_unnormalizer
from diffusers import DDPMScheduler
import tqdm


class Agent:
    def __init__(self, loaded_policy=None, action_type="continous"):
        self.loaded_policy = loaded_policy
        self.action_type = action_type

    def select_action(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

class RandomAgent(Agent):
    def __init__(self, num_actions, action_type="continous", momentum_length=None):
        super().__init__(action_type=action_type)
        #self.action_space = action_space # boundaries of the action space if continous; num of actions in discrete
        self.num_actions = num_actions
        self.momentum_length = momentum_length
        self.actions_in_a_row = 0
        self.last_action = None

    def select_action(self, observation):
        # if self.action_type == "continous":
        #     x_action = np.random.uniform(self.action_space[0][0], self.action_space[0][1])
        #     y_action = np.random.uniform(self.action_space[1][0], self.action_space[1][1])
        #     return (x_action, y_action)
        # else:
        if self.actions_in_a_row < self.momentum_length and self.last_action is not None:
            self.actions_in_a_row += 1
            return self.last_action
        else:
            self.actions_in_a_row = 0
            self.last_action = np.random.choice(self.num_actions)
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
    
class DiffusorAgent(Agent):
    def __init__(self, target, num_actions, path_to_diffusor_model):
        super().__init__()
        self.target = target
        self.num_actions = num_actions
        self.path_to_diffusor_model = path_to_diffusor_model
        self.model = self.load_diffusor_model()

        # ----------------- #
        # Hyperparameters
        # ----------------- #
        self.learning_rate = 1e-4
        self.eta = 1.0
        self.batch_size = 1
        self.min = 0
        self.max = 20
        # ----------------- #

    def select_action(self, observation, current_timestep): # Observation = (x: float, y: float) #TODO: Need to updated select_action method from other agents to work with third argument
        start_unnormalized = # Current position and action of the agent
        target_unnormalized = self.target
        start_normalized = # Normalize the start position and action
        target_normalized = # Normalize the target position
        horizon = MISSION_LENGTH - current_timestep# Remaining time until episode ends
        trajectory_normalized = self.get_trajectory_for_given_start_target_horizon(start_normalized, target_normalized, horizon)
        trajectory_unnormalized = # Unnormalize the trajectory
        next_action = # Get the next action from the trajectory
        return next_action

    def load_diffusor_model(self):
        model = get_model("unet1d")
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        model, optimizer = load_checkpoint(model, optimizer, self.path_to_diffusor_model)
        return model

    def get_trajectory_for_given_start_target_horizon(self, start, target, horizon):
        conditions = {
                    0: start,
                    -1: target
                }
        shape = (1,self.state_dim+self.action_dim, horizon)
        x = torch.randn(shape, device='cpu')
        x = reset_start_and_target(x, conditions, self.action_dim)
        scheduler = DDPMScheduler(num_train_timesteps=self.num_train_timesteps, prediction_type="sample")

        for i in tqdm.tqdm(scheduler.timesteps):

            timesteps = torch.full((self.batch_size,), i, device='cpu', dtype=torch.long)

            with torch.no_grad():
                # print("shape of x and timesteps: ", x.shape, timesteps.shape)
                residual = self.model(x, timesteps).sample

            obs_reconstruct = scheduler.step(residual, i, x)["prev_sample"]

            if self.eta > 0:
                noise = torch.randn(obs_reconstruct.shape).to(obs_reconstruct.device)
                posterior_variance = scheduler._get_variance(i)
                obs_reconstruct = obs_reconstruct + int(i>0) * (0.5 * posterior_variance) * self.eta* noise  # no noise when t == 0

            obs_reconstruct_postcond = reset_start_and_target(obs_reconstruct, conditions, self.action_dim)
            x = obs_reconstruct_postcond

        return x

# make a new agent called DrunkenAgent which by chance acts either like the RandomAgent or the NaiveAgent

class DrunkenAgent(Agent):
    def __init__(self, target, num_actions, action_type="discrete", momentum_length=3, magnitude=1):
        super().__init__(action_type=action_type)
        # Initialize RandomAgent and NaiveAgent objects
        self.last_agent_type = "naive"
        self.random_agent = RandomAgent(num_actions=num_actions, momentum_length=momentum_length,
                                        action_type=action_type)
        self.naive_agent = NaiveAgent(target=target, num_actions=num_actions, action_type=action_type,
                                      magnitude=magnitude)
    
    def select_action(self, observation):
        # Randomly choose between the two agents
        if self.random_agent.actions_in_a_row >= 1:
            self.last_agent_type = "random"
            return self.random_agent.select_action(observation)
        elif np.random.uniform() < 0.5:
            self.last_agent_type = "random"
            return self.random_agent.select_action(observation)
        else:
            self.last_agent_type = "naive"
            return self.naive_agent.select_action(observation)


    


# class CriticNetwork(nn.Module):
#     def __init__(self):
#         super(CriticNetwork, self).__init__()
#         # Define the layers directly without a Sequential wrapper
#         self.layer0 = nn.Linear(in_features=2, out_features=64)
#         self.layer1 = nn.Tanh()
#         self.layer2 = nn.Linear(in_features=64, out_features=64)
#         self.layer3 = nn.Tanh()
#         self.layer4 = nn.Linear(in_features=64, out_features=8)
#         self.layer5 = nn.Identity()

#     def forward(self, x):
#         x = self.layer0(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.layer5(x)
#         return x

class CQLAgent(Agent):
    def __init__(self, observation_shape, num_actions, num_layers, hidden_size, loaded_policy=None, action_type="discrete"):
        assert loaded_policy is not None, "Please specify path to CQL"
        self.loaded_policy = loaded_policy
        self.action_type = action_type
        self.critic = ptu.build_mlp(
            input_size=np.prod(observation_shape),
            output_size=num_actions,
            n_layers=num_layers,
            size=hidden_size,
        )
        self.num_actions = num_actions
        self.critic.load_state_dict(torch.load(self.loaded_policy))
        

    def select_action(self, observation: np.ndarray, epsilon: float = 0.04) -> int:
        observation = ptu.from_numpy(np.asarray(observation))[None]

        # TODO(student): get the action from the critic using an epsilon-greedy strategy
        """
        action = ...
        """
        if torch.rand(1) < epsilon:
            action = torch.randint(self.num_actions, ())
        else:
            qa_values: torch.Tensor = self.critic(observation)
            action = qa_values.argmax(dim=-1)
        # ENDTODO

        return ptu.to_numpy(action).squeeze(0).item()
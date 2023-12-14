import numpy as np
import random
import torch.nn as nn
import torch
import pytorch_util as ptu

from utils_diffusor import load_checkpoint, get_model, reset_start_and_target, limits_normalizer, limits_unnormalizer
from diffusers import DDPMScheduler
from value_planner import RNNValueNetwork 
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
    def __init__(self, target, num_actions, path_to_diffusor_model, MISSION_LENGTH, path_to_guided_model, GUIDED):
        super().__init__()

        # ----------------- #
        # Hyperparameters
        # ----------------- #

        self.learning_rate = 1e-4
        self.eta = 1.0
        self.batch_size = 3
        self.min = 0
        self.max = 20

        # ----------------- #

        self.state_dim = 2
        self.action_dim = 2
        self.num_train_timesteps = 3
        self.target = target
        self.num_actions = num_actions
        self.path_to_diffusor_model = path_to_diffusor_model
        self.path_to_guided_model = path_to_guided_model
        self.model = self.load_diffusor_model()
        self.MISSION_LENGTH = MISSION_LENGTH
        self.GUIDED = GUIDED
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


        if self.GUIDED:
            self.value_model = self.load_value_model(path_to_guided_model)



    def temp_normalizer(self, x):
        '''
        Normalizes the input tensor to the range [-1, 1].
        Assumes x is a PyTorch tensor.
        '''
        print("Normalizing data according to limits: ")
        print("x.min(): ", self.min)
        print("x.max(): ", self.max)
        x_normalized = 2 * (x - self.min) / (self.max - self.min) - 1
        return x_normalized



    def reshape_and_unpack(self, x):
        batch_size, num_features, horizon = x.shape
        unpacked_trajectories = []

        for b in range(batch_size):
            trajectory = np.empty((horizon, num_features)) 

            for t in range(horizon):
                action_state = x[b, :, t].cpu().numpy()
                trajectory[t, :] = action_state

            unpacked_trajectories.append(trajectory)

        return unpacked_trajectories



    def load_value_model(self, checkpoint_path):
        value_model = RNNValueNetwork(4, 256, 1, 2)  # Adjust parameters as necessary
        checkpoint = torch.load(checkpoint_path, map_location=self.DEVICE)
        
        if "model_state_dict" in checkpoint:
            value_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            value_model.load_state_dict(checkpoint)

        value_model.to(self.DEVICE)
        value_model.eval() 
        return value_model



    def select_action(self, observation, current_timestep): # Observation = (x: float, y: float) #TODO: Need to updated select_action method from other agents to work with third argument
     
        # Convert observation and target to tensors
        start_unnormalized = torch.tensor([observation[0], observation[1]], dtype=torch.float32)
        target_unnormalized = torch.tensor([self.target[0], self.target[1]], dtype=torch.float32)

        # Normalize the tensors
        start_normalized = self.temp_normalizer(start_unnormalized)
        target_normalized = self.temp_normalizer(target_unnormalized)
        
        horizon = self.MISSION_LENGTH  # Remaining time until episode ends
        # horizon = self.MISSION_LENGTH - current_timestep # Remaining time until episode ends
        trajectory_normalized = self.get_trajectory_for_given_start_target_horizon(start_normalized, target_normalized, horizon)

        trajectory_unnormalized = limits_unnormalizer(trajectory_normalized.cpu(), self.min, self.max)
        next_action = (trajectory_unnormalized[0,0, 0].item(), trajectory_unnormalized[0, 1, 0].item())  # Get the next action from the trajectory
        return next_action


    def load_diffusor_model(self):
        model = get_model("unet1d")
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        model, optimizer = load_checkpoint(model, optimizer, self.path_to_diffusor_model)
        return model

    def get_trajectory_for_given_start_target_horizon(self, start, target, horizon):
        device = next(self.model.parameters()).device  # Get the device of the model

        conditions = {
            0: start.to(device),
            -1: target.to(device)
        }
        shape = (1, self.state_dim + self.action_dim, horizon)

        x = torch.randn(shape, device=device)  # Ensure tensor is on the same device as model
        x = reset_start_and_target(x, conditions, self.action_dim)
        scheduler = DDPMScheduler(num_train_timesteps=self.num_train_timesteps, prediction_type="sample")

        for i in tqdm.tqdm(scheduler.timesteps):
            timesteps = torch.full((self.batch_size,), i, device=device, dtype=torch.long)

            with torch.no_grad():
                # print("Shape of x:", x.shape)
                # print("Shape of timesteps:", timesteps.shape)

                residual = self.model(x, timesteps).sample

            obs_reconstruct = scheduler.step(residual, i, x)["prev_sample"]

            if self.eta > 0:
                noise = torch.randn(obs_reconstruct.shape).to(device)
                posterior_variance = scheduler._get_variance(i)
                obs_reconstruct = obs_reconstruct + int(i > 0) * (0.5 * posterior_variance) * self.eta * noise

            obs_reconstruct_postcond = reset_start_and_target(obs_reconstruct, conditions, self.action_dim)
            x = obs_reconstruct_postcond



            if self.GUIDED and i % 2 == 0:
                print("\n doing value guided")
                trajectories = self.reshape_and_unpack(x)
                trajectory_tensors = [torch.tensor(traj, dtype=torch.float32, device=device) for traj in trajectories]

                with torch.no_grad():
                    values = torch.stack([self.value_model(traj_tensor.unsqueeze(0)) for traj_tensor in trajectory_tensors]).squeeze()

                # Select the trajectory with the highest estimated reward
                best_trajectory_index = torch.argmax(values).item()
                x = x[best_trajectory_index].unsqueeze(0).repeat(self.batch_size, 1, 1)

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
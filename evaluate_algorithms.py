from agent import CQLAgent, NaiveAgent, DiffusorAgent
from simulator import Environment, SingleGyreFlowField
import numpy as np
import matplotlib.pyplot as plt

from utils_custom import create_random_coordinate

LOADED_POLICY = "working_cql_implementation.pth"
POLICY_NAME = "VALUE_GUIDED"
PLOT_NAME = "TEST_VG"
VALUE_GUIDED = True


path_to_diffusor_model = '/home/bearhaon/CustomDiffusor/models/13-12-2023_23-43-01_new_100_final_step_2700.ckpt'
path_to_guided_model = '/home/bearhaon/CustomDiffusor/models/32_new_value_model_final.ckpt'


FLOW_FIELD = SingleGyreFlowField(width=20, height=20, center=(10, 10), radius=4, strength=1)

ACTION_TYPE = "continous"  # continous or discrete
NUM_ACTIONS = 8
MAGNITUDE = 1 # Magnitude of Action

MAX_STEPS = 32 # MUST BE MULTIPLE OF 8 FOR DIFFUSER

MISSION_STARTS = [[4.0,4.0], [2.0,8.0], [8.0,2.0], [8.0,8.0]]
MISSION_START_BOUNDS = [(1, 4.5), (1, 4.5)]
MISSION_TARGET = [17.5, 17.5]
THRESHOLD = 1.0
NUM_MISSIONS = 100


RENDER = True

overview = {"Naive": [{} for _ in range(NUM_MISSIONS)],
            "CQL": [{} for _ in range(NUM_MISSIONS)]}

def check_success(target, last_coordinate, threshold):
    # Calculate the distance between the agent and the target
    distance = ((last_coordinate[0] - target[0]) ** 2 + (last_coordinate[1] - target[1]) ** 2) ** 0.5
    reached_goal = (distance <= threshold)
    return reached_goal


def calculate_cumulative_reward(env):
    return np.array(env.reward_history).sum()

def calculate_ultimate_distance(env):
    return ((env.current_state[0] - env.target[0]) ** 2 + (env.current_state[1] - env.target[1]) ** 2) ** 0.5


success = []
cumulative_reward = []
ultimate_distance = []

for i in range(NUM_MISSIONS):

    start = create_random_coordinate(MISSION_START_BOUNDS[0], MISSION_START_BOUNDS[1])
    print("Start:", start)
    target = MISSION_TARGET
    print("Target:", target)

    #agent = NaiveAgent(target, NUM_ACTIONS, magnitude=MAGNITUDE)
    # agent = CQLAgent(loaded_policy=LOADED_POLICY, observation_shape=(2,),
    #                  num_actions=NUM_ACTIONS, num_layers=2, hidden_size=64, action_type=ACTION_TYPE)


    # INSERT DIFF AGENT
    agent = DiffusorAgent(target = target, num_actions=NUM_ACTIONS, path_to_diffusor_model=path_to_diffusor_model, MISSION_LENGTH = MAX_STEPS, path_to_guided_model = path_to_guided_model, GUIDED = VALUE_GUIDED)


    env = Environment(FLOW_FIELD, list(start), target, threshold=THRESHOLD,
                        action_type=ACTION_TYPE, num_actions=NUM_ACTIONS, magnitude=MAGNITUDE,
                        save_render_name="Naive" + str(i))
    state = env.reset()
    done = False
    step = 0
    while not done and step < MAX_STEPS:
        # PASS IN STEP AS I
        action = agent.select_action(state, step)
        new_state, reward, done = env.step(action)
        state = new_state
        step += 1


    success.append(check_success(target, env.current_state, THRESHOLD))
    cumulative_reward.append(calculate_cumulative_reward(env))
    ultimate_distance.append(calculate_ultimate_distance(env))

      
    fig, ax = env.render()
    if i <= 6:
        fig.savefig(f"trajectory_plots/evaluation/{PLOT_NAME}_{i}.png")
        plt.close(fig)

    #print("Number of successes: ", np.array([overview["CQL"][j]["success"] for j in range(NUM_MISSIONS)]).astype(int).sum())
    print("Number of successes: ", np.array(success).astype(int).sum())
    print("Average cumulative reward: ", np.array(cumulative_reward).mean())
    print("Average ultimate distance: ", np.array(ultimate_distance).mean())








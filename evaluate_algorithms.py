from agent import CQLAgent, NaiveAgent
from simulator import Environment, SingleGyreFlowField
import numpy as np
import matplotlib.pyplot as plt

LOADED_POLICY = "working_cql_implementation.pth"

FLOW_FIELD = SingleGyreFlowField(width=20, height=20, center=(10, 10), radius=4, strength=1)

ACTION_TYPE = "discrete"
NUM_ACTIONS = 8
MAGNITUDE = 1 # Magnitude of Action

MAX_STEPS = 100

MISSION_STARTS = [[4.0,4.0], [2.0,8.0], [8.0,2.0], [8.0,8.0]]
MISSION_TARGET = [17.5, 17.5]
THRESHOLD = 1.0
NUM_MISSIONS = len(MISSION_STARTS)

RENDER = True

overview = {"Naive": [{} for _ in range(NUM_MISSIONS)],
            "CQL": [{} for _ in range(NUM_MISSIONS)]}

def check_success(target, last_coordinate, threshold):
    # Calculate the distance between the agent and the target
    distance = ((last_coordinate[0] - target[0]) ** 2 + (last_coordinate[1] - target[1]) ** 2) ** 0.5
    reached_goal = (distance <= threshold)
    return reached_goal


for i, start in enumerate(MISSION_STARTS):
    print("Start:", start)
    target = MISSION_TARGET
    print("Target:", target)

    #agent = NaiveAgent(target, NUM_ACTIONS, magnitude=MAGNITUDE)
    agent = CQLAgent(loaded_policy=LOADED_POLICY, observation_shape=(2,),
                     num_actions=NUM_ACTIONS, num_layers=2, hidden_size=64, action_type=ACTION_TYPE)
    env = Environment(FLOW_FIELD, list(start), target, threshold=THRESHOLD,
                        action_type=ACTION_TYPE, num_actions=NUM_ACTIONS, magnitude=MAGNITUDE,
                        save_render_name="Naive" + str(i))
    state = env.reset()
    done = False
    step = 0
    while not done and step < MAX_STEPS:
        action = agent.select_action(state)
        new_state, reward, done = env.step(action)
        state = new_state
        step += 1

    # overview["Naive"][i]["start"] = start
    # overview["Naive"][i]["target"] = target
    # overview["Naive"][i]["trajectory"] = env.history
    # overview["Naive"][i]["last_coordinate"] = env.current_state
    # overview["Naive"][i]["success"] = check_success(target, env.current_state, THRESHOLD)
    overview["CQL"][i]["start"] = start
    overview["CQL"][i]["target"] = target
    overview["CQL"][i]["trajectory"] = env.history
    overview["CQL"][i]["last_coordinate"] = env.current_state
    overview["CQL"][i]["success"] = check_success(target, env.current_state, THRESHOLD)

    fig, ax = env.render()
    fig.savefig(f"trajectory_plots/evaluation/CQL_newest_all_traj{i}.png")
    print("esel")
    plt.close(fig)

print("Number of successes: ", np.array([overview["CQL"][j]["success"] for j in range(NUM_MISSIONS)]).astype(int).sum())


    




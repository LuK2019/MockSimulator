from agent import NaiveAgent
from simulator import Environment, SingleGyreFlowField
import numpy as np


FLOW_FIELD = SingleGyreFlowField(width=20, height=20, center=(10, 10), radius=4, strength=1)

ACTION_TYPE = "discrete"
NUM_ACTIONS = 8
MAGNITUDE = 1 # Magnitude of Action

MAX_STEPS = 100

MISSION_STARTS = [[2,2], [2,8], [8,2], [8,8]]
MISSION_ENDS = [[8,8], [8,2], [2,8], [2,2]]
THRESHOLD = 1.0
NUM_MISSIONS = len(MISSION_STARTS)

RENDER = True

overview = {"Naive": [{} for _ in range(NUM_MISSIONS)]}

def check_success(target, last_coordinate, threshold):
    # Calculate the distance between the agent and the target
    distance = ((last_coordinate[0] - target[0]) ** 2 + (last_coordinate[1] - target[1]) ** 2) ** 0.5
    reached_goal = (distance <= threshold)
    return reached_goal


for i, (start, target) in enumerate(zip(MISSION_STARTS, MISSION_ENDS)):
    print("Start:", start)
    print("Target:", target)

    agent = NaiveAgent(target, NUM_ACTIONS, magnitude=MAGNITUDE)
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

    overview["Naive"][i]["start"] = start
    overview["Naive"][i]["target"] = target
    overview["Naive"][i]["trajectory"] = env.history
    overview["Naive"][i]["last_coordinate"] = env.current_state
    overview["Naive"][i]["success"] = check_success(target, env.current_state, THRESHOLD)

    if RENDER:
        env.render()

print(np.array([overview["Naive"][j]["success"] for j in range(NUM_MISSIONS)]).astype(int).sum())


    




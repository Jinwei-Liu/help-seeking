import numpy as np
from tqdm import tqdm
import random
from scipy.stats import entropy
import pandas as pd
from sarsa import Sarsa
from draw_result import draw_result
from grid_world_env import CustomFrozenLake, maps

# Fix random seed
seed = 0
np.random.seed(seed)
random.seed(seed)

# Parameters
params = {
    'selected_map': 1,
    'num_episodes': 200,
    'state_dim': 9 * 9,
    'action_dim': 4,
    'epsilon_decay': 0.9,
    'epsilon_min': 0,
    'epsilon_decay_threshold': 160,
    'slippery_positions_map_1': [(4, 2), (4, 3), (4, 4), (4, 5), (4, 6)]
}

# Select map
desc_def = maps[params['selected_map']]
agent = Sarsa(params['state_dim'], params['action_dim'])

# Specify slippery positions for map 1
slippery_positions = params['slippery_positions_map_1'] if params['selected_map'] == 1 else []

env = CustomFrozenLake(desc=desc_def, slippery_positions=slippery_positions, selected_map=params['selected_map'])

return_list = []
for i in range(params['num_episodes']):
    agent.epsilon *= params['epsilon_decay']
    if i > params['epsilon_decay_threshold']:
        agent.epsilon = params['epsilon_min']
    print('episodes', agent.epsilon)
    with tqdm(total=1000, desc='Iteration %d' % i) as pbar:
        for i_episode in range(1000):
            episode_return = 0
            state, info = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                next_action = agent.take_action(next_state)
                done = terminated or truncated
                agent.update(state, action, reward, next_state, next_action, done)
                state = next_state
                episode_return += reward
            return_list.append(episode_return)
            if (i_episode + 1) % 200 == 0:
                pbar.set_postfix({
                    'episode': '%d' % (params['num_episodes'] / 10 * i + i_episode + 1),
                    'return': '%.3f' % np.mean(return_list[-200:])
                })
            pbar.update(1)

var_Q = agent.M - agent.Q ** 2
position = np.argmax(agent.Q, axis=1)

draw_result(agent.Q, position, "grid_world/result/Q_values_{}.jpg".format(params['selected_map']))
draw_result(var_Q, position, "grid_world/result/Var_{}.jpg".format(params['selected_map']))

# Save Q-values and variance to Excel files
q_df = pd.DataFrame(agent.Q)
var_q_df = pd.DataFrame(var_Q)
q_df.to_excel(f"grid_world/result/Q_values_{params['selected_map']}.xlsx", index=True)
var_q_df.to_excel(f"grid_world/result/Var_{params['selected_map']}.xlsx", index=True)

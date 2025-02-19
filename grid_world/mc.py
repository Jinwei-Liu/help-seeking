import pandas as pd
import numpy as np
from grid_world_env import CustomFrozenLake, maps
# Fix random seed
seed = 0
np.random.seed(seed)

# Read Q-values and variance for map 0 and map 1
Q_map0 = pd.read_excel("grid_world/result/Q_values_0.xlsx", index_col=0, engine='openpyxl').to_numpy()
var_Q_map0 = pd.read_excel("grid_world/result/Var_0.xlsx", index_col=0, engine='openpyxl').to_numpy()
Q_map1 = pd.read_excel("grid_world/result/Q_values_1.xlsx", index_col=0, engine='openpyxl').to_numpy()
var_Q_map1 = pd.read_excel("grid_world/result/Var_1.xlsx", index_col=0, engine='openpyxl').to_numpy()

def select_action(Q, state):
    Q_list = Q[state, :]
    Q_list = (Q_list + 1e-9) / (Q_list + 1e-9).sum()
    return np.random.choice(4, p=Q_list)

def monte_carlo_estimation(env, Q, num_episodes=50000, gamma=0.995):
    state_returns = {state: [] for state in range(env.observation_space.n)}
    for _ in range(num_episodes):
        state, info = env.reset()
        episode = []
        done = False
        while not done:
            action = select_action(Q, state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode.append((state, reward))
            state = next_state
        G = 0
        for state, reward in reversed(episode):
            G = reward + gamma * G
            state_returns[state].append(G)
    return {state: np.var(returns) for state, returns in state_returns.items() if returns}

def run_monte_carlo(selected_map, Q_map, var_Q_map):
    desc_def = maps[selected_map]
    slippery_positions = [(4, 2), (4, 3), (4, 4), (4, 5), (4, 6)] if selected_map == 1 else []
    env = CustomFrozenLake(desc=desc_def, slippery_positions=slippery_positions, selected_map=selected_map)
    var_reward_map = monte_carlo_estimation(env, Q_map)
    array = np.zeros(81)
    for key, value in var_reward_map.items():
        array[key] = value
    max_indices = np.argmax(Q_map, axis=1)
    result = [var_Q_map[i, idx] for i, idx in enumerate(max_indices)]
    print(np.mean(np.abs(result - array)))

# Run Monte Carlo estimation for both maps
run_monte_carlo(0, Q_map0, var_Q_map0)
run_monte_carlo(1, Q_map1, var_Q_map1)
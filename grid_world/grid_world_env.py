from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
import numpy as np

class CustomFrozenLake(FrozenLakeEnv):
    def __init__(self, desc, slippery_positions, selected_map):
        super().__init__(desc=desc, is_slippery=False)  # , render_mode='human'
        self.slippery_positions = slippery_positions  # Positions that should be slippery
        self.selected_map = selected_map

    def step(self, action):
        # Get current position
        row, col = self.s // self.ncol, self.s % self.ncol

        # Check if in slippery area
        if (row, col) in self.slippery_positions:
            # Simulate slipping behavior
            if self.np_random.uniform(0, 1) < 0.05:
                action = self.np_random.choice(4)  # Randomly choose an action
        next_state, reward, terminated, truncated, info = super().step(action)

        if self.selected_map == 0:
            if next_state == 26:
                reward = np.random.normal(10, 5)
            elif next_state == 62:
                reward = np.random.normal(10, 0)
        else:
            reward *= 10

        # Call the original step method
        return (next_state, reward, terminated, truncated, info)

# Map definitions
maps = {
    0: [
        "FFFFHHHHH",
        "FFFFHHHHH",
        "FFFFFFFFG",
        "FFFFHHHHH",
        "SFFFHHHHH",
        "FFFFHHHHH",
        "FFFFFFFFG",
        "FFFFHHHHH",
        "FFFFHHHHH",
    ],
    1: [
        "FFFFFFFFF",
        "FFHHHHHFF",
        "FFHHHHHFF",
        "FFHHHHHFF",
        "SFFFFFFFG",
        "FFHHHHHFF",
        "FFHHHHHFF",
        "FFHHHHHFF",
        "FFFFFFFFF",
    ]
}

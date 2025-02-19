## Files

- `sarsa_grid_world.py`: Main script to run the SARSA algorithm on the grid world environment.
- `grid_world_env.py`: Custom grid world environment based on FrozenLake.
- `sarsa.py`: Implementation of the SARSA algorithm.
- `draw_result.py`: Utility to visualize the results.
- `MC.py`: Script to run Monte Carlo estimation for variance of rewards.
  
## Usage

1. **Run the training script:**
    ```sh
    python sarsa_grid_world.py
    ```

2. **Run the Monte Carlo estimation script:**
    ```sh
    python MC.py
    ```

3. **View the results:**
    The results will be saved as images and Excel files in the `grid_world/result` directory.

## Customization

- **Maps:**
    You can switch between different maps by changing the `selected_map` variable in `sarsa_grid_world.py`.

- **Slippery Positions:**
    Modify the `slippery_positions` variable in `sarsa_grid_world.py` to specify which positions should be slippery.

## Example

To run the SARSA algorithm on the default map with slippery positions, simply execute the training script:

```sh
python sarsa_grid_world.py
```

To run the Monte Carlo estimation for variance of rewards, execute the following script:

```sh
python MC.py
```

## License

This project is licensed under the MIT License.
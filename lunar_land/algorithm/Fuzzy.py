import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

def fuzzy_inference(strategy_uncertainty_value, reward_uncertainty_value):

    # 定义输入变量
    strategy_uncertainty = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'strategy_uncertainty')
    reward_uncertainty = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'reward_uncertainty')

    # 定义输出变量
    output = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'output')

    # 定义隶属函数（模糊集合）
    strategy_uncertainty['low'] = fuzz.trapmf(strategy_uncertainty.universe, [0, 0, 0.3, 0.5])
    strategy_uncertainty['medium'] = fuzz.trimf(strategy_uncertainty.universe, [0.3, 0.5, 0.7])
    strategy_uncertainty['high'] = fuzz.trapmf(strategy_uncertainty.universe, [0.5, 0.7, 1, 1])

    reward_uncertainty['low'] = fuzz.trapmf(reward_uncertainty.universe, [0, 0, 0.3, 0.5])
    reward_uncertainty['medium'] = fuzz.trimf(reward_uncertainty.universe, [0.3, 0.5, 0.7])
    reward_uncertainty['high'] = fuzz.trapmf(reward_uncertainty.universe, [0.5, 0.7, 1, 1])

    output['low'] = fuzz.trapmf(output.universe, [0, 0, 0.3, 0.5])
    output['medium'] = fuzz.trimf(output.universe, [0.3, 0.5, 0.7])
    output['high'] = fuzz.trapmf(output.universe, [0.5, 0.7, 1, 1])

    # 定义模糊规则
    rule1 = ctrl.Rule(strategy_uncertainty['low'] & reward_uncertainty['low'], output['low'])
    rule2 = ctrl.Rule(strategy_uncertainty['low'] & reward_uncertainty['medium'], output['medium'])
    rule3 = ctrl.Rule(strategy_uncertainty['low'] & reward_uncertainty['high'], output['high'])
    rule4 = ctrl.Rule(strategy_uncertainty['medium'] & reward_uncertainty['low'], output['high'])
    rule5 = ctrl.Rule(strategy_uncertainty['medium'] & reward_uncertainty['medium'], output['high'])
    rule6 = ctrl.Rule(strategy_uncertainty['medium'] & reward_uncertainty['high'], output['high'])
    rule7 = ctrl.Rule(strategy_uncertainty['high'] & reward_uncertainty['low'], output['high'])
    rule8 = ctrl.Rule(strategy_uncertainty['high'] & reward_uncertainty['medium'], output['high'])
    rule9 = ctrl.Rule(strategy_uncertainty['high'] & reward_uncertainty['high'], output['high'])

    # 创建控制系统
    system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
    sim = ctrl.ControlSystemSimulation(system)

    # 输入值
    sim.input['strategy_uncertainty'] = strategy_uncertainty_value
    sim.input['reward_uncertainty'] = reward_uncertainty_value

    # 运行模糊推理
    sim.compute()

    # 输出结果
    output_value = sim.output['output']

    return output_value

def visualize_relationship():
    strategy_uncertainty = np.arange(0, 1.1, 0.1)
    reward_uncertainty = np.arange(0, 1.1, 0.1)
    output_values = np.zeros((len(strategy_uncertainty), len(reward_uncertainty)))

    for i, su in enumerate(strategy_uncertainty):
        for j, ru in enumerate(reward_uncertainty):
            output_values[i, j] = fuzzy_inference(su, ru)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(strategy_uncertainty, reward_uncertainty)
    ax.plot_surface(X, Y, output_values.T, cmap='viridis')
    ax.set_xlabel('Strategy Uncertainty', fontsize=12)
    ax.set_ylabel('Reward Uncertainty', fontsize=12)
    ax.set_zlabel('Output', fontsize=12)
    ax.set_title('Relationship between Uncertainties and Output', fontsize=15)
    ax.view_init(elev=30, azim=-110)  
    plt.savefig('relationship.png', dpi=1000)
    # plt.show()

if __name__ == "__main__":
    # 调用示例
    result = fuzzy_inference(0.7, 0.8)
    print(f"推理输出值: {result}")

    result = fuzzy_inference(0.1, 0.4)
    print(f"推理输出值: {result}")

    result = fuzzy_inference(0.0, 0.8)
    print(f"推理输出值: {result}")

    result = fuzzy_inference(0.35, 0.0)
    print(f"推理输出值: {result}")

    visualize_relationship()

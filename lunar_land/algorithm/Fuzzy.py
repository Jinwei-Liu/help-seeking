import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

class FuzzyInferenceSystem:
    def __init__(self):
        # 定义输入变量
        self.strategy_uncertainty = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'strategy_uncertainty')
        self.reward_uncertainty = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'reward_uncertainty')

        # 定义输出变量
        self.output = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'output')

        # 定义隶属函数（模糊集合）
        self.strategy_uncertainty['low'] = fuzz.trapmf(self.strategy_uncertainty.universe, [0, 0, 0.3, 0.5])
        self.strategy_uncertainty['medium'] = fuzz.trimf(self.strategy_uncertainty.universe, [0.3, 0.5, 0.9])
        self.strategy_uncertainty['high'] = fuzz.trapmf(self.strategy_uncertainty.universe, [0.5, 0.7, 1, 1])

        self.reward_uncertainty['low'] = fuzz.trapmf(self.reward_uncertainty.universe, [0, 0, 0.3, 0.5])
        self.reward_uncertainty['medium'] = fuzz.trimf(self.reward_uncertainty.universe, [0.3, 0.5, 0.7])
        self.reward_uncertainty['high'] = fuzz.trapmf(self.reward_uncertainty.universe, [0.5, 0.7, 1, 1])

        self.output['low'] = fuzz.trapmf(self.output.universe, [0, 0, 0.3, 0.6])
        self.output['medium'] = fuzz.trimf(self.output.universe, [0.5, 0.5, 0.8])
        self.output['high'] = fuzz.trapmf(self.output.universe, [0.7, 0.9, 1, 1])

        # 定义模糊规则
        rule1 = ctrl.Rule(self.strategy_uncertainty['low'] & self.reward_uncertainty['low'], self.output['low'])
        rule2 = ctrl.Rule(self.strategy_uncertainty['low'] & self.reward_uncertainty['medium'], self.output['medium'])
        rule3 = ctrl.Rule(self.strategy_uncertainty['low'] & self.reward_uncertainty['high'], self.output['high'])
        rule4 = ctrl.Rule(self.strategy_uncertainty['medium'] & self.reward_uncertainty['low'], self.output['medium'])
        rule5 = ctrl.Rule(self.strategy_uncertainty['medium'] & self.reward_uncertainty['medium'], self.output['high'])
        rule6 = ctrl.Rule(self.strategy_uncertainty['medium'] & self.reward_uncertainty['high'], self.output['high'])
        rule7 = ctrl.Rule(self.strategy_uncertainty['high'] & self.reward_uncertainty['low'], self.output['high'])
        rule8 = ctrl.Rule(self.strategy_uncertainty['high'] & self.reward_uncertainty['medium'], self.output['high'])
        rule9 = ctrl.Rule(self.strategy_uncertainty['high'] & self.reward_uncertainty['high'], self.output['high'])

        # 创建控制系统
        self.system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
        self.sim = ctrl.ControlSystemSimulation(self.system)

    def infer(self, strategy_uncertainty_value, reward_uncertainty_value):
        # 输入值
        self.sim.input['strategy_uncertainty'] = strategy_uncertainty_value
        self.sim.input['reward_uncertainty'] = reward_uncertainty_value

        # 运行模糊推理
        self.sim.compute()

        # 输出结果
        return self.sim.output['output']

# 创建全局的 FuzzyInferenceSystem 实例
fis = FuzzyInferenceSystem()

def fuzzy_inference(strategy_uncertainty_value, reward_uncertainty_value):
    return fis.infer(strategy_uncertainty_value, reward_uncertainty_value)

def visualize_relationship():
    """ Generate a 3D plot to visualize the relationship between uncertainties and the fuzzy output. """
    strategy_uncertainty = np.linspace(0, 1, 11)
    reward_uncertainty = np.linspace(0, 1, 11)
    output_values = np.zeros((len(strategy_uncertainty), len(reward_uncertainty)))

    # Compute fuzzy output for the entire range
    for i, su in enumerate(strategy_uncertainty):
        for j, ru in enumerate(reward_uncertainty):
            output_values[i, j] = fuzzy_inference(su, ru)

    # Generate 3D surface plot
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(strategy_uncertainty, reward_uncertainty)
    
    ax.plot_surface(X, Y, output_values.T, cmap='viridis', edgecolor='gray', alpha=0.7)
    ax.set_xlabel('Strategy Uncertainty', fontsize=12)
    ax.set_ylabel('Reward Uncertainty', fontsize=12)
    ax.set_zlabel('Fuzzy Output', fontsize=12)
    # ax.set_title('Fuzzy Inference System Output', fontsize=15)
    ax.view_init(elev=30, azim=-120)
    
    # Improve plot aesthetics
    ax.grid(True)
    plt.savefig('fuzzy_relationship.png', dpi=300)
    plt.show()

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

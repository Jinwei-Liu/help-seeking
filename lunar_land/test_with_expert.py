from algorithm.TD3 import one_hot_encode, Action_adapter
from algorithm.Fuzzy import fuzzy_inference
import os
from random import choice
import matplotlib.pyplot as plt
from LunarLander_env import *
from tqdm import tqdm
import random
from collections import deque
import statsmodels.api as sm
import pandas as pd
def normalize_value(value, min_val, max_val):

    if value < min_val:
        return 0
    elif value > max_val:
        return 1
    else:
        return (value - min_val) / (max_val - min_val)

def is_triggered_by_reward_var_val(reward_var_val, threshold):
    reward_var_val = normalize_value(reward_var_val, 0, 12000)
    return reward_var_val > threshold

def is_triggered_by_a_variance(a_variance, threshold):
    return a_variance[0] > threshold

def is_triggered(a_variance, reward_var_val, threshold):
    reward_var_val = normalize_value(reward_var_val, 0, 12000)
    total_uncertain = fuzzy_inference(a_variance[0], reward_var_val)
    return total_uncertain > threshold


def collect_test_data(use_expert, triggerway, threshold, env, agent, expert_agent, opt, episodes, noise_level=0):

    rewards_per_episode = []  # 用于存储每个回合的总奖励
    call_times = 0
    call_station = {0: [0,0], 1: [0,0], 2: [0,0]}
    for _ in tqdm(range(episodes), desc="Progress {} {}".format(noise_level, threshold)):  # 使用 tqdm 显示测试进度条
        station = 0
        call_station[station][0] += 1
        aim_position = random.randint(1, 9)
        aim_vector = one_hot_encode(aim_position - 1, 9)
        aim_vector_with_noise = aim_vector.copy()

        if aim_position==1 or aim_position==9:
            noise_position = np.random.choice([1, 9])
            aim_vector_with_noise[aim_position - 1] = 0
            aim_vector_with_noise[noise_position - 1] = 1

        elif aim_position==5:
            noise_position = np.random.choice([1, 9])
            aim_vector_with_noise[noise_position - 1] = 1

        # 重置环境并设置初始状态
        s, info = env.reset(land_position=aim_position, difficult_mode=opt.difficult_mode)
        # 将噪声目标向量与状态拼接
        s_with_noise = np.append(s, aim_vector_with_noise)
        # 将原始目标向量与状态拼接
        s_without_noise = np.append(s, aim_vector)

        # 初始化存储动作和方差的数组
        a = np.zeros((100, 2))  # 用于存储 100 次动作
        var = np.zeros((100, 1))  # 用于存储每次动作对应的方差

        total_reward = 0  # 初始化总奖励

        done = False  # 标志回合是否结束
        calling_expert = False

        # 用于平滑
        a_variances = deque([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        reward_var_vals = deque([[0], [0], [0], [0], [0]])

        while not done:  # 开始每个回合的循环
            for i in range(100):  # 生成 100 次动作
                # 使用强化学习智能体选择动作及对应的方差
                a[i], var[i] = agent.select_action_with_var(s_with_noise)

            # 计算 100 次动作的均值和方差
            a_mean = np.mean(a, axis=0)  # 动作均值
            a_variance = np.var(a, axis=0)  # 动作方差
            a_variances.append(a_variance)
            a_variances.popleft() 
            
            reward_var_val = np.mean(var, axis=0)  # 方差的平均值
            reward_var_vals.append(reward_var_val)
            reward_var_vals.popleft() 

            # 通过专家智能体选择动作
            a_expert = expert_agent.select_action(s_without_noise)

            # 调整动作的范围以适配环境
            act = Action_adapter(a_mean, opt.max_action)
            act_expert = Action_adapter(a_expert, opt.max_action)

            if not calling_expert and use_expert and 0.05<s_without_noise[1]<0.5:
                if triggerway == 0:
                    calling_expert = is_triggered_by_reward_var_val(np.mean(reward_var_vals), threshold)
                elif triggerway == 1:
                    calling_expert = is_triggered_by_a_variance(np.mean(a_variances, axis=0), threshold)
                else:
                    calling_expert = is_triggered(np.mean(a_variances, axis=0), np.mean(reward_var_vals), threshold)

            # 判断是否使用专家智能体的动作
            if calling_expert and use_expert:  # 如果动作方差超过阈值
                s_next, r, dw, tr, info = env.step(act_expert)  # 使用专家动作与环境交互
                call_station[station][1] += 1
                # 直接关闭给390的奖励
                total_reward = 390
                break

            else:
                s_next, r, dw, tr, info = env.step(act)  # 使用强化学习智能体的动作与环境交互

            total_reward += r  # 累加当前步的奖励

            # 更新下一步状态（带噪声和不带噪声）
            s_next_with_noise = np.append(s_next, aim_vector_with_noise)
            s_next_without_noise = np.append(s_next, aim_vector)

            # 更新回合结束标志（dw 表示正常结束，tr 表示提前结束）
            done = (dw or tr)

            # 更新当前状态为下一步的状态
            s_with_noise = s_next_with_noise + np.random.normal(0, noise_level, s_next_with_noise.shape)
            s_without_noise = s_next_without_noise

        if calling_expert:
            call_times += 1
        # 将当前回合的总奖励添加到列表中
        rewards_per_episode.append(total_reward)

    return rewards_per_episode, call_times, call_station # 返回 奖励列表 表示测试完成

def save_data(output_folder, triggerway, thresholds, mean_results, var_results, call_expert_times, rewards_by_triggerway):
    """保存收集的数据到文件"""
    np.save(os.path.join(output_folder, f'mean_results_{triggerway}.npy'), mean_results[triggerway])
    np.save(os.path.join(output_folder, f'var_results_{triggerway}.npy'), var_results[triggerway])
    np.save(os.path.join(output_folder, f'call_expert_times_{triggerway}.npy'), call_expert_times[triggerway])
    for threshold in thresholds:
        np.save(os.path.join(output_folder, f'rewards_{triggerway}_{threshold:.2f}.npy'), rewards_by_triggerway[triggerway][threshold])

def load_data(output_folder, triggerway, thresholds):
    """从文件加载收集的数据"""
    mean_results = np.load(os.path.join(output_folder, f'mean_results_{triggerway}.npy')).tolist()
    var_results = np.load(os.path.join(output_folder, f'var_results_{triggerway}.npy')).tolist()
    call_expert_times = np.load(os.path.join(output_folder, f'call_expert_times_{triggerway}.npy')).tolist()
    rewards_by_triggerway = {threshold: np.load(os.path.join(output_folder, f'rewards_{triggerway}_{threshold:.2f}.npy')).tolist() for threshold in thresholds}
    return mean_results, var_results, call_expert_times, rewards_by_triggerway

def plot_results(output_folder, triggerways, thresholds, mean_results, var_results, call_expert_times, rewards_by_triggerway):
    """绘制学术风格的散点图（第一张图）"""
    plt.figure(figsize=(8, 6), dpi=300)  

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]  
    markers = ["o", "s", "D", "^", "v"]  

    for i, triggerway in enumerate(triggerways):
        x = np.array(call_expert_times[triggerway])
        y = np.array(mean_results[triggerway])

        # 按 x 轴排序
        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]

        # 移动平均平滑
        df = pd.DataFrame({"x": x_sorted, "y": y_sorted})
        df["y_smooth"] = df["y"].rolling(window=10, center=True, min_periods=1).mean()

        # 绘制原始散点
        plt.scatter(x, y, alpha=0.6, color=colors[i % len(colors)], 
                    edgecolors="black", marker=markers[i % len(markers)], 
                    label=f"Triggerway {triggerway}")

        # 绘制平滑曲线
        plt.plot(df["x"], df["y_smooth"], color=colors[i % len(colors)], 
                 linestyle="-", linewidth=2.5, label=f"Smoothed {triggerway}")

    # 学术风格标题 & 坐标轴
    plt.title(r"Mean Return vs. Call Expert Times", fontsize=14, fontweight="bold")
    plt.xlabel(r"Call Expert Times", fontsize=12)
    plt.ylabel(r"Mean Return", fontsize=12)

    # 网格优化
    plt.grid(True, linestyle="--", alpha=0.6)

    # 调整刻度字体
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # 图例放置右上角，带透明背景
    plt.legend(fontsize=10, loc="lower right", frameon=True, facecolor="white", framealpha=0.8)

    # 保存第一张图
    save_path = os.path.join(output_folder, "mean_vs_call_expert_scatter.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    
    print(f"图像已保存至 {save_path}")

    # ---------------------------------------
    # **第二张图（均值和方差的误差棒图）**
    # ---------------------------------------
    plt.figure(figsize=(12, 6), dpi=300)  

    for i, triggerway in enumerate(triggerways):
        # 计算标准差作为误差棒
        std_dev = np.sqrt(var_results[triggerway])

        # 误差棒图
        plt.errorbar(thresholds, mean_results[triggerway], yerr=std_dev, 
                     fmt='-o', color=colors[i % len(colors)], marker=markers[i % len(markers)], 
                     capsize=4, capthick=1.5, elinewidth=1.5, alpha=0.9, linewidth=2, 
                     label=f'Triggerway {triggerway}')

    # 学术风格标题 & 轴标签
    plt.title(r"Mean and Variance of Rewards for Different Triggerways", fontsize=14, fontweight="bold")
    plt.xlabel(r"Threshold", fontsize=12)
    plt.ylabel(r"Reward", fontsize=12)

    # 网格优化
    plt.grid(True, linestyle="--", alpha=0.6)

    # 调整刻度字体
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # **调整图例（右下角，带透明背景）**
    plt.legend(fontsize=10, loc="lower right", frameon=True, facecolor="white", framealpha=0.8)

    # 保存第二张图
    save_path2 = os.path.join(output_folder, "mean_and_variance_summary_with_errorbars.png")
    plt.savefig(save_path2, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"图像已保存至 {save_path2}")
    
def test_with_expert(env, agent, expert_agent, opt, episodes=1000, use_saved_data=True, noise_level=0):
    # 定义保存图像的文件夹
    output_folder = os.path.join('result', f'noise_{noise_level}')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 触发方式和阈值组合
    triggerways = [0, 1, 2]
    thresholds = np.arange(0.0, 1.02, 0.01)  # 阈值从 0 到 1，步长为 0.02

    # 记录 mean 和 variance 的字典
    mean_results = {0: [], 1: [], 2: []}  # 记录每个 triggerway 的 mean
    var_results = {0: [], 1: [], 2: []}   # 记录每个 triggerway 的 variance
    call_expert_times = {0: [], 1: [], 2: []}   # 记录每个 triggerway 的 variance

    # 记录每个 triggerway 和 threshold 的 rewards，用于绘制图表
    rewards_by_triggerway = {0: {th: [] for th in thresholds}, 
                            1: {th: [] for th in thresholds}, 
                            2: {th: [] for th in thresholds}}

    if use_saved_data:
        try:
            for triggerway in triggerways:
                mean_results[triggerway], var_results[triggerway], call_expert_times[triggerway], rewards_by_triggerway[triggerway] = load_data(output_folder, triggerway, thresholds)
            print("Successfully loaded saved data from", output_folder)
        except FileNotFoundError:
            print("No saved data found. Collecting new data...")
            use_saved_data = False

    if not use_saved_data:
        for triggerway in triggerways:
            for threshold in thresholds:
                # 收集测试数据
                reward_with_expert, call_times, call_station = collect_test_data(True, triggerway, threshold, env, agent, expert_agent, opt, episodes=episodes, noise_level=noise_level)
                # 计算 mean 和 variance
                with_expert_mean = np.mean(reward_with_expert)
                with_expert_variance = np.var(reward_with_expert)

                # 保存 mean 和 variance 到对应的字典
                mean_results[triggerway].append(with_expert_mean)
                var_results[triggerway].append(with_expert_variance)
                call_expert_times[triggerway].append(call_times)

                # 打印统计信息
                print(f"Triggerway {triggerway}, Threshold {threshold:.2f}:")
                print(f"With Expert - Mean: {with_expert_mean:.4f}, Variance: {with_expert_variance:.4f}, Call_times: {call_times}")
                print("Call_station:", call_station)

                # 将当前 reward_with_expert 添加到对应的 triggerway 和 threshold 组合中
                rewards_by_triggerway[triggerway][threshold] = reward_with_expert

            # 保存收集的数据
            save_data(output_folder, triggerway, thresholds, mean_results, var_results, call_expert_times, rewards_by_triggerway)

    # 调用绘图函数
    plot_results(output_folder, triggerways, thresholds, mean_results, var_results, call_expert_times, rewards_by_triggerway)



def collect_test_data_fast(env, agent, opt, episodes, noise_level=0):
    all_states = []
    all_rewards = []
    all_a_variances = []
    all_reward_var_vals = []
    for _ in tqdm(range(episodes), desc="Progress {}".format(noise_level)):  # 使用 tqdm 显示测试进度条
        episode_states = []
        episode_a_variances = []
        episode_reward_var_vals = []
        aim_position = random.randint(1, 9)
        aim_vector = one_hot_encode(aim_position - 1, 9)
        aim_vector_with_noise = aim_vector.copy()

        if aim_position==1 or aim_position==9:
            noise_position = np.random.choice([1, 9])
            aim_vector_with_noise[aim_position - 1] = 0
            aim_vector_with_noise[noise_position - 1] = 1

        elif aim_position==5:
            noise_position = np.random.choice([1, 9])
            aim_vector_with_noise[noise_position - 1] = 1

        # 重置环境并设置初始状态
        s, info = env.reset(land_position=aim_position, difficult_mode=opt.difficult_mode)
        # 将噪声目标向量与状态拼接
        s_with_noise = np.append(s, aim_vector_with_noise)

        # 初始化存储动作和方差的数组
        a = np.zeros((100, 2))  # 用于存储 100 次动作
        var = np.zeros((100, 1))  # 用于存储每次动作对应的方差

        total_reward = 0  # 初始化总奖励

        done = False  # 标志回合是否结束

        # 用于平滑
        a_variances = deque([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        reward_var_vals = deque([[0], [0], [0], [0], [0]])

        while not done:  # 开始每个回合的循环
            for i in range(100):  # 生成 100 次动作
                # 使用强化学习智能体选择动作及对应的方差
                a[i], var[i] = agent.select_action_with_var(s_with_noise)

            # 计算 100 次动作的均值和方差
            a_mean = np.mean(a, axis=0)  # 动作均值
            a_variance = np.var(a, axis=0)  # 动作方差
            a_variances.append(a_variance)
            a_variances.popleft() 
            
            reward_var_val = np.mean(var, axis=0)  # 方差的平均值
            reward_var_vals.append(reward_var_val)
            reward_var_vals.popleft() 

            # 调整动作的范围以适配环境
            act = Action_adapter(a_mean, opt.max_action)

            s_next, r, dw, tr, info = env.step(act)  # 使用强化学习智能体的动作与环境交互

            total_reward += r  # 累加当前步的奖励

            # 记录当前状态、a_variance 和 reward_var_val
            episode_states.append(s_next.tolist())
            episode_a_variances.append(np.mean(a_variances, axis=0).tolist())
            episode_reward_var_vals.append(np.mean(reward_var_vals).tolist())

            # 更新下一步状态（带噪声和不带噪声）
            s_next_with_noise = np.append(s_next, aim_vector_with_noise)

            # 更新回合结束标志（dw 表示正常结束，tr 表示提前结束）
            done = (dw or tr)

            # 更新当前状态为下一步的状态
            s_with_noise = s_next_with_noise + np.random.normal(0, noise_level, s_next_with_noise.shape)

        all_states.append(episode_states)
        all_rewards.append(total_reward)
        all_a_variances.append(episode_a_variances)
        all_reward_var_vals.append(episode_reward_var_vals)

    return all_states, all_rewards, all_a_variances, all_reward_var_vals

def test_with_expert_fast(env, agent, opt, episodes=1000, use_saved_data=True, noise_level=0):
    # 定义保存图像的文件夹
    output_folder = os.path.join('result_fast', f'noise_{noise_level}')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if use_saved_data:
        try:
            all_states = np.load(os.path.join(output_folder, 'all_states.npy'), allow_pickle=True).tolist()
            all_rewards = np.load(os.path.join(output_folder, 'all_rewards.npy'), allow_pickle=True).tolist()
            all_a_variances = np.load(os.path.join(output_folder, 'all_a_variances.npy'), allow_pickle=True).tolist()
            all_reward_var_vals = np.load(os.path.join(output_folder, 'all_reward_var_vals.npy'), allow_pickle=True).tolist()
            print("Successfully loaded saved data from", output_folder)
        except FileNotFoundError:
            print("No saved data found. Collecting new data...")
            use_saved_data = False

    if not use_saved_data:
        all_states, all_rewards, all_a_variances, all_reward_var_vals = collect_test_data_fast(env, agent, opt, episodes=episodes, noise_level=noise_level)
        np.save(os.path.join(output_folder, 'all_states.npy'), np.array(all_states, dtype=object))
        np.save(os.path.join(output_folder, 'all_rewards.npy'), np.array(all_rewards, dtype=object))
        np.save(os.path.join(output_folder, 'all_a_variances.npy'), np.array(all_a_variances, dtype=object))
        np.save(os.path.join(output_folder, 'all_reward_var_vals.npy'), np.array(all_reward_var_vals, dtype=object))
    
    # 触发方式和阈值组合
    triggerways = [0, 1, 2]
    thresholds = np.arange(0.0, 1.02, 0.01)  # 阈值从 0 到 1，步长为 0.02

    # 记录 mean 和 variance 的字典
    mean_results = {0: [], 1: [], 2: []}  # 记录每个 triggerway 的 mean
    var_results = {0: [], 1: [], 2: []}   # 记录每个 triggerway 的 variance
    call_expert_times = {0: [], 1: [], 2: []}   # 记录每个 triggerway 的 variance

    # 记录每个 triggerway 和 threshold 的 rewards，用于绘制图表
    rewards_by_triggerway = {0: {th: [] for th in thresholds}, 
                            1: {th: [] for th in thresholds}, 
                            2: {th: [] for th in thresholds}}

    for triggerway in triggerways:
        for threshold in thresholds:
            call_times = 0
            rewards_per_episode = []
            for episode_index in range(episodes):
                episode_states = all_states[episode_index]
                episode_a_variances = all_a_variances[episode_index]
                episode_reward_var_vals = all_reward_var_vals[episode_index]
                total_reward = all_rewards[episode_index]
                
                calling_expert = False
                for step_index in range(len(episode_states)):
                    state = episode_states[step_index]
                    a_variance = episode_a_variances[step_index]
                    reward_var_val = episode_reward_var_vals[step_index]

                    # Mimic the expert assistance logic
                    if not calling_expert and 0.05 < state[1] < 0.5:
                        if triggerway == 0:
                            calling_expert = is_triggered_by_reward_var_val(reward_var_val, threshold)
                        elif triggerway == 1:
                            calling_expert = is_triggered_by_a_variance(a_variance, threshold)
                        else:
                            calling_expert = is_triggered(a_variance,reward_var_val, threshold)
                    
                    if calling_expert:
                        call_times += 1
                        total_reward = 390
                        break
                
                rewards_per_episode.append(total_reward)
            
            # 计算 mean 和 variance
            with_expert_mean = np.mean(rewards_per_episode)
            with_expert_variance = np.var(rewards_per_episode)

            # 保存 mean 和 variance 到对应的字典
            mean_results[triggerway].append(with_expert_mean)
            var_results[triggerway].append(with_expert_variance)
            call_expert_times[triggerway].append(call_times)

            # 打印统计信息
            print(f"Triggerway {triggerway}, Threshold {threshold:.2f}:")
            print(f"With Expert - Mean: {with_expert_mean:.4f}, Variance: {with_expert_variance:.4f}, Call_times: {call_times}")

            # 将当前 rewards_per_episode 添加到对应的 triggerway 和 threshold 组合中
            rewards_by_triggerway[triggerway][threshold] = rewards_per_episode

    # 调用绘图函数
    plot_results(output_folder, triggerways, thresholds, mean_results, var_results, call_expert_times, rewards_by_triggerway)

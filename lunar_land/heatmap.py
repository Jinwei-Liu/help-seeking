from algorithm.TD3 import one_hot_encode, Action_adapter
import matplotlib.pyplot as plt
from LunarLander_env import *
from tqdm import tqdm
import os
import random

def get_aim_and_noise_position(form):
    """获取不同形式下的目标位置和噪声位置"""
    if form == 1:  # 正常情况
        aim_position = random.randint(1, 9)
        noise_position = aim_position
    elif form == 2:  # 被干扰
        aim_position = 5
        noise_position = np.random.choice([1, 9])
    # elif form == 3:  # 目标丢失
    #     aim_position = 5
    #     noise_position = aim_position
    else:
        raise ValueError("Invalid form number")
    return aim_position, noise_position

def get_aim_vector(aim_position, noise_position, form):
    """获取不同形式下的目标向量"""
    aim_vector = one_hot_encode(aim_position - 1, 9)
    # if form == 3:  # 目标丢失情况
    #     aim_vector[noise_position-1] = 0
    # else:  # 形式1和形式2
    aim_vector[noise_position-1] = 1
    return aim_vector

def collect_action_var_data(env, agent, opt, episodes, grid_size, x_range, y_range, form=1):

    # 初始化存储动作方差数据的网格和计数器
    x_vals = np.zeros(grid_size)  # 存储 x 方向动作方差累积值
    y_vals = np.zeros(grid_size)  # 存储 y 方向动作方差累积值
    reward_var_vals = np.zeros(grid_size)  # 存储回报方差累积值
    count = np.zeros(grid_size)   # 存储每个网格被访问的次数

    x_min, x_max = x_range  # 获取 x 范围
    y_min, y_max = y_range  # 获取 y 范围

    # 多次运行环境以收集数据
    for _ in tqdm(range(episodes), desc="Progress"):
        reward_total = 0

        # 根据形式获取目标位置和噪声位置
        aim_position, noise_position = get_aim_and_noise_position(form)
        aim_vector = get_aim_vector(aim_position, noise_position, form)

        s, info = env.reset(land_position=aim_position, noise_position=noise_position, difficult_mode=opt.difficult_mode)
        s = np.append(s, aim_vector)  # 将目标向量拼接到状态向量中

        a = np.zeros((100, 2))  # 存储 100 次动作的数组
        var = np.zeros((100, 1))  # 存储每次动作的方差

        done = False
        while not done:
            x, y = s[0], s[1]  # 提取当前状态的 x 和 y 坐标

            for i in range(100):
                # 使用智能体选择动作及其方差
                a[i], var[i] = agent.select_action_with_var(s)
            
            # 计算动作的均值和方差
            a_mean = np.mean(a, axis=0)  # 平均动作值
            a_variance = np.var(a, axis=0)  # 动作方差
            reward_var_val = np.mean(var, axis=0)

            # 将动作转换为环境可接受的范围
            act = Action_adapter(a_mean, opt.max_action)

            # 执行动作并获取下一个状态、奖励和其他信息
            s_next, r, dw, tr, info = env.step(act)
            reward_total += r
            s_next = np.append(s_next, aim_vector)
            done = (dw or tr)  # 判断当前回合是否结束
            s = s_next

            # 将 (x, y) 坐标映射到离散网格
            x_idx = int((x - x_min) / (x_max - x_min) * (grid_size[0] - 1))
            y_idx = int((y - y_min) / (y_max - y_min) * (grid_size[1] - 1))

            # 确保网格索引在合法范围内
            x_idx = np.clip(x_idx, 0, grid_size[0] - 1)
            y_idx = np.clip(y_idx, 0, grid_size[1] - 1)

            # 累加动作方差到对应网格中
            x_vals[x_idx, y_idx] += a_variance[0]  # 累加 x 方向方差
            y_vals[x_idx, y_idx] += a_variance[1]  # 累加 y 方向方差

            reward_var_vals[x_idx, y_idx] += reward_var_val[0]
            
            count[x_idx, y_idx] += 1  # 增加访问计数
        
    # 计算平均方差，避免除以零的情况
    action_variance_x = np.divide(x_vals, count, where=(count != 0))
    action_variance_x[count == 0] = 0
    action_variance_y = np.divide(y_vals, count, where=(count != 0))
    action_variance_y[count == 0] = 0
    reward_variance = np.divide(reward_var_vals, count, where=(count != 0))
    reward_variance[count == 0] = 0

    return action_variance_x, action_variance_y, reward_variance

from matplotlib.colors import LinearSegmentedColormap

def plot_heatmap(heatmap, x_range, y_range, save_address, vmin=0, vmax=1):
    # 创建自定义颜色映射
    end_color = np.array([22, 104, 151]) / 255.0  # 转换为 0-1 范围
    start_color = np.array([255, 255, 255]) / 255.0   # 转换为 0-1 范围
    
    # 创建包含白色作为起始颜色的颜色映射
    colors = ['white', start_color, end_color]
    # 设置颜色停止点，使得0值为白色
    stops = [0, 0.01, 1]  # 0值为白色，略大于0开始使用渐变色
    
    n_bins = 100  # 颜色渐变的平滑度
    custom_cmap = LinearSegmentedColormap.from_list('custom', list(zip(stops, colors)), N=n_bins)

    # 创建网格坐标轴范围
    x_ticks = np.linspace(x_range[0], x_range[1], heatmap.shape[0])
    y_ticks = np.linspace(y_range[0], y_range[1], heatmap.shape[1])

    # 设置图像样式
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'

    # 创建图像
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 绘制热力图
    im = ax.imshow(heatmap.T, cmap=custom_cmap, origin='lower',
                   extent=[x_ticks.min(), x_ticks.max(), y_ticks.min(), y_ticks.max()],
                   vmin=vmin, vmax=vmax)
    
    # # 添加颜色条并设置样式
    # cbar = plt.colorbar(im, ax=ax, label="Average Variance")
    # cbar.ax.set_ylabel("Average Variance", fontsize=10)
    plt.colorbar(im, ax=ax)
    
    # 设置标题和轴标签
    ax.set_title("Average Variance", fontsize=12, pad=10)
    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel("y", fontsize=10)
    
    # 保存图片
    plt.savefig(save_address, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def draw_action_var_heatmap(env, agent, opt, save_address, use_saved_data=False, episodes=1000, grid_size=(100, 100), x_range=(-1, 1), y_range=(-0.1, 1.5), form=2):
    # 根据不同形式创建对应的文件夹名
    form_name = f"form{form}"
    output_folder = os.path.join('heatmap', form_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if use_saved_data:
        try:
            # 尝试加载已保存的数据
            action_variance_x = np.load(os.path.join(output_folder, f'action_variance_x_form{form}.npy'))
            action_variance_y = np.load(os.path.join(output_folder, f'action_variance_y_form{form}.npy'))
            reward_variance = np.load(os.path.join(output_folder, f'reward_variance_form{form}.npy'))
            print(f"Successfully loaded saved data for form {form} from {output_folder}")
        except FileNotFoundError:
            print("No saved data found. Collecting new data...")
            use_saved_data = False
    
    if not use_saved_data:
        # 收集新的动作方差数据
        action_variance_x, action_variance_y, reward_variance = collect_action_var_data(
            env, agent, opt, episodes, grid_size, x_range, y_range, form=form)

        # 保存 heatmap 数据
        np.save(os.path.join(output_folder, f'action_variance_x_form{form}.npy'), action_variance_x)
        np.save(os.path.join(output_folder, f'action_variance_y_form{form}.npy'), action_variance_y)
        np.save(os.path.join(output_folder, f'reward_variance_form{form}.npy'), reward_variance)
        print(f"New data collected and saved to {output_folder}")

    # 绘制热力图，文件名中添加形式标识
    save_address_a1 = save_address + f"action_x_form{form}.png"
    plot_heatmap(action_variance_x, x_range, y_range, save_address_a1)
    save_address_a2 = save_address + f"action_y_form{form}.png"
    plot_heatmap(action_variance_y, x_range, y_range, save_address_a2)
    save_address_rewrd = save_address + f"reward_form{form}.png"
    plot_heatmap(reward_variance, x_range, y_range, save_address_rewrd, vmin=0, vmax=12000)

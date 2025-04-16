import numpy as np
import matplotlib.pyplot as plt
import os

def draw_call_expert_hist(data_folder, output_folder):
    """
    绘制柱状图，展示不同 noise level 下，不同触发方式第一次达到 380 奖励时的求助次数的均值和方差。

    Args:
        data_folder (str): 包含数据的文件夹路径。
        output_folder (str): 保存图像的文件夹路径。
    """

    noise_levels = [0.0, 0.3, 0.6, 0.9]
    triggerways = [0, 1, 2]
    num_noise_levels = len(noise_levels)
    num_triggerways = len(triggerways)
    bar_width = 0.8 / num_triggerways  # 调整柱的宽度，使它们不重叠

    # 设置颜色
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    labels = ["RUBR", "PNUR", "HILHS-TD3(ours)"]

    # 创建柱状图
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    x = np.arange(num_noise_levels)  # x 轴是 noise levels

    for i, triggerway in enumerate(triggerways):
        # 提取每个 noise level 下，所有 run 的第一次超过 380 分数的求援次数
        call_expert_means = []
        call_expert_stds = []

        for noise_idx in range(num_noise_levels):
            noise_level = noise_levels[noise_idx]
            
            # 读取所有 run 中，triggerway 对应的 rewards_by_triggerway
            all_runs_call_expert_times = []
            for run_idx in range(10):  # 假设有 5 个 runs
                try:

                    file_path = os.path.join(data_folder, f'noise_{noise_level}', f'mean_results_run_{run_idx}.npy')
                    rewards = np.load(file_path, allow_pickle=True).item()

                    file_path = os.path.join(data_folder, f'noise_{noise_level}', f'call_expert_times_run_{run_idx}.npy')
                    call_times = np.load(file_path, allow_pickle=True).item()

                    call_expert_times = 0
                    for j, rewad in enumerate(rewards[triggerway]):
                        if rewad <= 380:
                            call_expert_times = call_times[triggerway][j]
                            break
                    all_runs_call_expert_times.append(call_expert_times)

                except FileNotFoundError:
                    print(f"File not found: {file_path}")
                    return
            
            # 计算平均 call expert 次数和标准差
            mean_call_expert_times = np.mean(all_runs_call_expert_times)
            std_call_expert_times = np.std(all_runs_call_expert_times)
            call_expert_means.append(mean_call_expert_times)
            call_expert_stds.append(std_call_expert_times)

        # 计算每个 triggerway 的柱的 x 轴位置
        positions = x + (i * bar_width) - 0.4 + bar_width / 2

        # 绘制柱状图
        ax.bar(positions, call_expert_means, width=bar_width, label=labels[i], color=colors[i % len(colors)], yerr=call_expert_stds, capsize=5)

    # 设置 x 轴刻度标签
    ax.set_xticks(x)
    ax.set_xticklabels(noise_levels)

    # 学术风格标题 & 坐标轴
    ax.set_title("Mean Call Expert Times vs. Noise Level", fontsize=14, fontweight="bold")
    ax.set_xlabel("Noise Level", fontsize=12)
    ax.set_ylabel("Mean Call Expert Times", fontsize=12)

    # 网格优化
    ax.grid(True, linestyle="--", alpha=0.6)

    # 调整刻度字体
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)

    # 图例放置右上角，带透明背景
    ax.legend(fontsize=10, loc="upper right", frameon=True, facecolor="white", framealpha=0.8)

    # 保存图像
    save_path = os.path.join(output_folder, "call_expert_times_histogram.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"图像已保存至 {save_path}")

if __name__ == '__main__':
    data_folder = 'result_fast'  # 替换为你的数据文件夹路径
    output_folder = 'result_hist'  # 替换为你的输出文件夹路径
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    draw_call_expert_hist(data_folder, output_folder)

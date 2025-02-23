# 修改噪声的产生方式
from algorithm.TD3 import TD3, str2bool, one_hot_encode, Action_adapter_reverse, Action_adapter, evaluate_policy
from datetime import datetime
import gym
import os, shutil
import argparse
import torch
import random
from random import choice
import torch.multiprocessing as mp
from LunarLander_env import *
from heatmap import *
from test_with_expert import *

def noise_postion_create(aim_position):
    # 以 30% 的概率修改 aim_vector
    if random.random() < 0.0:
        # 随机选择一个索引并将其设置为 1
        index_to_modify = random.randint(1, 9)
        return index_to_modify
    else:
        return aim_position

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0, help='CustomLunarLander-v2')

parser.add_argument('--mode', type=int, default=2 , help='0：Training 1：draw_heat_map 2:call_expert')
parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')

parser.add_argument('--difficult_mode', type=str2bool, default=True, help='difficult_mode or Not')

parser.add_argument('--Loadmodel', type=str2bool, default=True, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=0, help='which model to load')

parser.add_argument('--test_with_noise', type=str2bool, default=False, help='test_with_noise or Not')
parser.add_argument('--test_noise_position', type=int, default=5, help='test_with_noise or Not')
parser.add_argument('--test_aim_position', type=int, default=1, help='Which postion to land for testing')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=int(5e7), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(1e6), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(2e3), help='Model evaluating interval, in steps.')
parser.add_argument('--update_every', type=int, default=50, help='Training Fraquency, in stpes')

parser.add_argument('--gamma', type=float, default=0.995, help='Discounted Factor')
parser.add_argument('--a_lr', type=float, default=1e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=1e-4, help='Learning rate of critic')
parser.add_argument('--m_lr', type=float, default=1e-4, help='Learning rate of M')
parser.add_argument('--batch_size', type=int, default=1024, help='batch_size of training')
parser.add_argument('--alpha', type=float, default=0.02, help='Entropy coefficient')
parser.add_argument('--adaptive_alpha', type=str2bool, default=False, help='Use adaptive_alpha or Not')

opt = parser.parse_args()

# Seed Everything
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(opt.seed)
random.seed(opt.seed)
print("Random Seed: {}".format(opt.seed))
opt.dvc = torch.device(opt.dvc) # from str to torch.device
print(opt)

EnvName = ['CustomLunarLander']

def main():

    # # Build Env
    if opt.render:
        env = gym.make(EnvName[opt.EnvIdex], continuous=True, render_mode = 'human')
        env.action_space.seed(opt.seed) # act = env.action_space.sample() 需要固定
    else:
        env = gym.make(EnvName[opt.EnvIdex], continuous=True) #, enable_wind=True
        env.action_space.seed(opt.seed)
    eval_env = gym.make(EnvName[opt.EnvIdex], continuous=True)
    opt.state_dim = env.observation_space.shape[0] + 9
    opt.action_dim = env.action_space.shape[0]
    opt.max_action = float(env.action_space.high[0])   #remark: action space【-max,max】
    opt.max_e_steps = env._max_episode_steps
    print(f'Env:{EnvName[opt.EnvIdex]}  state_dim:{opt.state_dim}  action_dim:{opt.action_dim}  '
          f'max_a:{opt.max_action}  min_a:{env.action_space.low[0]}  max_e_steps:{opt.max_e_steps}')

    # Build SummaryWriter to record training curves
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}'.format(EnvName[opt.EnvIdex]) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    # Build DRL model
    if not os.path.exists('model_TD3'): os.mkdir('model_TD3')
    agent = TD3(**vars(opt)) # var: transfer argparse to dictionary

    if opt.Loadmodel: agent.load("./model_TD3/{}_{}_{}.pth".format(EnvName[opt.EnvIdex], opt.ModelIdex, "difficult" if opt.difficult_mode else "easy"))

    if opt.mode == 0:
        print("training")
        # 训练阶段
        agent.actor.train()
        total_steps = 0
        while total_steps < opt.Max_train_steps:
            aim_position = random.randint(1,9)
            aim_vector = one_hot_encode(aim_position-1,9)
            noise_position = noise_postion_create(aim_position)
            aim_vector[noise_position-1] = 1
            s, info = env.reset(land_position=aim_position, noise_position=noise_position, difficult_mode=opt.difficult_mode)# 最好不要提供随机种子否则容易过拟合
            s = np.append(s,aim_vector)
            done = False

            '''Interact & trian'''
            while not done:
                if total_steps < (5*opt.max_e_steps):
                    act = env.action_space.sample()  # act∈[-max,max]
                    a = Action_adapter_reverse(act, opt.max_action)  # a∈[-1,1]
                else:
                    a = agent.select_action(s)  # a∈[-1,1]
                    act = Action_adapter(a, opt.max_action)  # act∈[-max,max]

                s_next, r, dw, tr, info = env.step(act)  # dw: dead&win; tr: truncated
                s_next = np.append(s_next, aim_vector)
                done = (dw or tr)
                agent.replay_buffer.add(s, a, r, s_next, dw)
 
                s = s_next
                total_steps += 1

                '''train if it's time'''
                # train 50 times every 50 steps rather than 1 training per step. Better!
                if (total_steps >= 2*opt.max_e_steps) and (total_steps % opt.update_every == 0):
                    for i in range(opt.update_every):
                        agent.train(writer, total_steps, i)

                '''record & log'''
                if total_steps % opt.eval_interval == 0:
                    ep_r = evaluate_policy(eval_env, agent, opt, turns=10)
                    if opt.write: writer.add_scalar('ep_r', ep_r, global_step=total_steps)
                    print(f'EnvName:{EnvName[opt.EnvIdex]}, Steps: {int(total_steps/1000)}k, Episode Reward:{ep_r}')

                '''save model'''
                if total_steps % opt.save_interval == 0:
                    agent.save("./model_TD3/{}_{}_{}.pth".format(EnvName[opt.EnvIdex], int(total_steps/1000), "difficult" if opt.difficult_mode else "easy"))
                    print('save model')

    elif opt.mode == 1:
        print("draw_heatmap")
        agent.actor.train()
        save_address = "./heatmap/{}_".format("difficult" if opt.difficult_mode else "easy")
        draw_action_var_heatmap(env, agent, opt, save_address, episodes=100)

    else:
        print("call_expert")
        mp.set_start_method('spawn')
        processes = []
        noise_levels = [0, 0.05, 0.1, 0.15, 0.2]
        signal_queue = mp.Queue(maxsize=10)
        for noise_level in noise_levels:
            p = mp.Process(target=worker_process, args=(opt, noise_level, signal_queue))
            p.start()
            processes.append(p)

        finished_processes = 0
        while finished_processes < len(noise_levels):
            finished_noise_level = signal_queue.get()
            print(f"Process with noise level {finished_noise_level} has finished.")
            finished_processes += 1

        for p in processes:
            p.terminate()
        for p in processes:
            p.join()
        
    env.close()
    eval_env.close()

def worker_process(opt, noise_level, signal_queue):
    env = gym.make(EnvName[opt.EnvIdex], continuous=True)
    agent = TD3(**vars(opt))
    agent.load("./model_TD3/{}_{}_{}.pth".format(EnvName[opt.EnvIdex], opt.ModelIdex, "difficult" if opt.difficult_mode else "easy"))
    expert_agent = TD3(**vars(opt))
    expert_agent.load("./model_expert/expert_agent.pth")

    test_with_expert(env, agent, expert_agent, opt, episodes=100, noise_level=noise_level)
    env.close()
    signal_queue.put(noise_level)


if __name__ == '__main__':
    main()


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import argparse 
import random

def one_hot_encode(k, n):
    one_hot_vector = [0] * n
    one_hot_vector[k] = 1
    return one_hot_vector

def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def Action_adapter(a,max_action):
    #from [-1,1] to [-max,max]
    return  a*max_action


def Action_adapter_reverse(act,max_action):
    #from [-max,max] to [-1,1]
    return act/max_action


# Actor网络定义
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, p=0.1):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 256)
        self.dropout1 = nn.Dropout(p)  # 添加第一个 Dropout 层
        self.layer2 = nn.Linear(256, 512)
        self.dropout2 = nn.Dropout(p)  # 添加第二个 Dropout 层
        self.layer3 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(p) 
        self.layer4 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.layer1(state))
        x = self.dropout1(x)  # 在激活函数后应用 Dropout
        x = torch.relu(self.layer2(x))
        x = self.dropout2(x)  # 在激活函数后应用 Dropout
        x = torch.relu(self.layer3(x))
        x = self.dropout3(x)
        x = self.max_action * torch.tanh(self.layer4(x))
        return x


# Critic网络定义
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.layer4(x)
        return x

# M网络定义
class M(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(M, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.layer4(x)
        return x

class TD3:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.actor = Actor(self.state_dim, self.action_dim, self.max_action).to(self.dvc)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.max_action).to(self.dvc)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.a_lr)

        self.critic1 = Critic(self.state_dim, self.action_dim).to(self.dvc)
        self.critic1_target = Critic(self.state_dim, self.action_dim).to(self.dvc)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.c_lr)

        self.critic2 = Critic(self.state_dim, self.action_dim).to(self.dvc)
        self.critic2_target = Critic(self.state_dim, self.action_dim).to(self.dvc)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.c_lr)

        self.m_var = M(self.state_dim, self.action_dim).to(self.dvc)
        self.m_var_target = M(self.state_dim, self.action_dim).to(self.dvc)
        self.m_var_target.load_state_dict(self.m_var.state_dict())
        self.m_var_optimizer = optim.Adam(self.m_var.parameters(), lr=self.m_lr)

        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=int(1e6), dvc=self.dvc)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state[np.newaxis,:]).to(self.dvc)
            a = self.actor(state)
        return a.cpu().numpy()[0]

    def select_action_with_var(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state[np.newaxis,:]).to(self.dvc)
            a = self.actor(state)
            Q = self.critic1(state,a)
            M = self.m_var(state,a)
            var = M - Q ** 2
        return a.cpu().numpy()[0], var.cpu().numpy()[0]

    def train(self, writer, total_steps, it, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):

        # Sample replay buffer
        s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)

        # Select action according to policy and add clipped noise
        noise = torch.normal(0, policy_noise, size=a.shape).clamp(-noise_clip, noise_clip).to(self.dvc)
        next_action = self.actor_target.forward(s_next)
        next_action_with_noise = (next_action + noise).clamp(-self.max_action, self.max_action)

        # Compute target Q value
        target_Q1 = self.critic1_target(s_next, next_action_with_noise)
        target_Q2 = self.critic2_target(s_next, next_action_with_noise)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = r + (~dw) * self.gamma * target_Q

         # Compute target M value
        target_MQ1 = self.critic1_target(s_next, next_action)
        target_MQ2 = self.critic2_target(s_next, next_action)
        target_M_Q = torch.min(target_MQ1, target_MQ2)
        target_M =  self.m_var_target(s_next, next_action)
        target_M = r ** 2 + (~dw) * 2 * r * self.gamma * target_M_Q + (~dw) * self.gamma ** 2 * target_M
        
        # Get current Q estimates
        current_Q1 = self.critic1(s, a)
        current_Q2 = self.critic2(s, a)

        current_M = self.m_var(s, a)

        # Compute critic loss
        critic1_loss = nn.functional.mse_loss(current_Q1, target_Q.detach())
        critic2_loss = nn.functional.mse_loss(current_Q2, target_Q.detach())

        M_loss = nn.functional.mse_loss(current_M, target_M.detach())

        # Optimize the critic
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        self.m_var_optimizer.zero_grad()
        M_loss.backward()
        self.m_var_optimizer.step()

        if it == 0:
            if self.write: writer.add_scalar('c1_loss', critic1_loss, global_step=total_steps)
            if self.write: writer.add_scalar('c2_loss', critic2_loss, global_step=total_steps)
            if self.write: writer.add_scalar('M_loss', M_loss, global_step=total_steps)

        # Delayed policy updates
        if it % policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic1(s, self.actor(s)).mean()

            if self.write and it==0: writer.add_scalar('a_loss', actor_loss, global_step=total_steps)
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.m_var.parameters(), self.m_var_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


    def save(self, filename):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'm_var': self.m_var.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'm_var_optimizer': self.m_var_optimizer.state_dict()
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.m_var.load_state_dict(checkpoint['m_var'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        self.m_var_optimizer.load_state_dict(checkpoint['m_var_optimizer'])


# Sample replay buffer
class ReplayBuffer():
    def __init__(self, state_dim, action_dim, max_size, dvc):
        self.max_size = max_size
        self.dvc = dvc
        self.ptr = 0
        self.size = 0

        self.s = torch.zeros((max_size, state_dim) ,dtype=torch.float,device=self.dvc)
        self.a = torch.zeros((max_size, action_dim) ,dtype=torch.float,device=self.dvc)
        self.r = torch.zeros((max_size, 1) ,dtype=torch.float,device=self.dvc)
        self.s_next = torch.zeros((max_size, state_dim) ,dtype=torch.float,device=self.dvc)
        self.dw = torch.zeros((max_size, 1) ,dtype=torch.bool,device=self.dvc)

    def add(self, s, a, r, s_next, dw):
        #每次只放入一个时刻的数据
        self.s[self.ptr] = torch.from_numpy(s).to(self.dvc)
        self.a[self.ptr] = torch.from_numpy(a).to(self.dvc) # Note that a is numpy.array
        self.r[self.ptr] = r
        self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.dvc)
        self.dw[self.ptr] = dw

        self.ptr = (self.ptr + 1) % self.max_size #存满了又重头开始存
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, device=self.dvc, size=(batch_size,))
        return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]



def evaluate_policy(env, agent, opt, turns=3):
    total_scores = 0
    for _ in range(turns):
        aim_position = random.randint(1, 9)
        aim_vector = one_hot_encode(aim_position-1,9)
        s, info = env.reset(land_position=aim_position, noise_position=aim_position, difficult_mode=opt.difficult_mode, seed=opt.seed)
        s = np.append(s,aim_vector)
        done = False
        while not done:
            # Take deterministic actions at test time
            a= agent.select_action(s)
            s_next, r, dw, tr, info = env.step(a)
            s_next = np.append(s_next, aim_vector)
            done = (dw or tr)
            total_scores += r
            s = s_next
    return int(total_scores/turns)
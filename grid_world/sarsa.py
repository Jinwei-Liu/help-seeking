import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Sarsa(): 
    def __init__(self, state_dim, action_dim, gamma=0.995, epsilon=0.9, lr=0.02):
        super(Sarsa, self).__init__() 

        self.Q = np.zeros((state_dim, action_dim))  
        self.M = np.zeros((state_dim, action_dim))

        self.gamma = gamma 
        self.epsilon = epsilon
        self.act_dim = action_dim
        self.lr = lr

    def predict(self, obs):
        Q_list = self.Q[obs, :]
        Q_list = (Q_list + 1e-9) / (Q_list + 1e-9).sum()
        action = np.random.choice(self.act_dim, p=Q_list) 
        return action  
    
    def take_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.act_dim)  
        else: 
            return self.predict(state)                

    def update(self, obs, action, reward, next_obs, next_act, done):

        current_Q = self.Q[obs, action]     
        if done:    
            target_Q = reward       
        else:      
            target_Q = reward + self.gamma * self.Q[next_obs, next_act]

        self.Q[obs, action] += self.lr * (target_Q - current_Q)


        current_var = self.M[obs, action]     
        if done:    
            target_var = reward ** 2      
        else:       
            target_var = reward ** 2 + 2 * reward * self.gamma * self.Q[next_obs, next_act] + self.gamma * self.gamma * self.M[next_obs, next_act]

        self.M[obs, action] += self.lr * (target_var - current_var)


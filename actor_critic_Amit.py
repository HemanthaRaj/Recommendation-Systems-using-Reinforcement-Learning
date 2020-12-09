# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:51:26 2020


"""

import sys
import torch  
import gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd

# hyperparameters
hidden_size = 64
learning_rate = 0.001

# Constants
gamma = 0.8
num_step = 100
max_episode = 100

class ActorCritic(nn.Module):
    def __init__(self, num_input, num_action, hidden_size, learning_rate=0.001):
        super(ActorCritic, self).__init__()

        self.num_action = num_action
        self.critic_linear1 = nn.Linear(num_input, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)

        self.actor_linear1 = nn.Linear(num_input, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, num_action)
    
    def forward(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        value = F.relu(self.critic_linear1(state))
        value = self.critic_linear2(value)
        
        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)

        return value, policy_dist
    
    
def a2c(env):
    num_input = env.observation_space.shape[0]
    num_output = env.action_space.n
    
    actor_critic = ActorCritic(num_input, num_output, hidden_size)
    ac_optimizer = optim.Adam(actor_critic.parameters(), learn_rate=learning_rate)

    tot_length = []
    #average_lengths = []
    tot_reward = []
    entropy_term = 0
    
    for episode in range(max_episode):
        reward = []
        value = []
        log_prob = 0

        state = env.reset()
        for step in range(num_step):
            value, policy_dist = actor_critic.forward(state)
            value = value.detach().numpy()[0,0]
            dist = policy_dist.detach().numpy() 

            action = np.random.choice(num_output, p=np.squeeze(dist))
            entropy = -np.sum(np.mean(dist) * np.log(dist))
            log_prob = torch.log(policy_dist.squeeze(0)[action])
            #define a step function for your recommender
            new_state, reward, done, _ = env.step(action)
            
            reward.append(reward)
            value.append(value)
            entropy_term += entropy
            log_prob.append(log_prob)
            state = new_state
            
            if done or step == num_step-1:
                Qval, _ = actor_critic.forward(new_state)
                Qval = Qval.detach().numpy()[0,0]
                tot_length.append(step)
                #average_lengths.append(np.mean(all_lengths[-10:]))
                if episode % 10 == 0:                    
                    sys.stdout.write("episode: {}, reward: {}, total length: {} \n".format(episode, np.sum(reward), step))
                break
            
            # Q values computation
        Qvals = np.zeros_like(value)
        for s in reversed(range(len(reward))):
            Qval = reward[s] + gamma * Qval
            Qvals[s] = Qval
  
        #actor critic update
        value = torch.FloatTensor(value)
        Qvals = torch.FloatTensor(Qvals)
        log_prob = torch.stack(log_prob)
        
        adv = Qvals - value
        actor_loss = (-log_prob * adv).mean()
        critic_loss = 0.5 * adv.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.005 * entropy_term

        ac_optimizer.zero_grad()
        ac_loss.backward()
        ac_optimizer.step()
        
        # Plot results
    smoothed_reward = pd.Series.rolling(pd.Series(tot_reward), 10).mean()
    smoothed_reward = [i for i in smoothed_reward]
    plt.plot(tot_reward)
    plt.plot(smoothed_reward)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

    plt.plot(tot_length)
    #plt.plot(average_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Episode length')
    plt.show()
    
    
env = gym.make("CartPole-v0")
a2c(env)    
            

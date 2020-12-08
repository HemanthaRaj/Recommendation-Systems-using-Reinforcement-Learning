import os
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as neural
import torch.optim as optimizer
import torch.nn.functional as F
from torch.distributions import Categorical
from itertools import count
cuda = 'cpu'
import matplotlib.pyplot as plt

class Actor(neural.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = neural.Linear(self.state_size, 128)
        self.linear2 = neural.Linear(128, 256)
        self.linear3 = neural.Linear(256, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution


class Critic(neural.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = neural.Linear(self.state_size, 128)
        self.linear2 = neural.Linear(128, 256)
        self.linear3 = neural.Linear(256, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value

def reward_calc(action, ratings):
    if action == 1:
        rec = 1
    if action == 0:
        rec = -1
    if ratings > 0.75:
        reward = rec * 1
    if ratings < 0.25:
        reward = rec * 1 * -1
    if ratings > 0.5:
        reward = rec * 1
    if ratings > 0.25:
        reward = rec * 1 * -1
    return reward

def return_calc(next_value, rewards, masks):
    nv = next_value
    gamma = 0.99
    ret_val = []
    for step in reversed(range(len(rewards))):
        nv = rewards[step] + gamma * nv * masks[step]
        ret_val.insert(0, nv)
    return ret_val

def trainIters(id, user_ratings, actor, critic, tot_iterations):
    actor_optimizer = optimizer.Adam(actor.parameters())
    critic_optimizer = optimizer.Adam(critic.parameters())
    r_graph = []
    print(user_ratings)
    # print('User ID: ' + str(id))
    for iter in range(tot_iterations):
        state = np.array([id, 0.0])
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0
        tot_reward = []
        for i in count():
            ratings = user_ratings[i]
            state = torch.FloatTensor(state).to(cuda)
            dist, value = actor(state), critic(state)
            action = dist.sample()
            reward = reward_calc(action, ratings)
            tot_reward.append(reward)
            if i == 99:
                done = True
            else:
                done = False
                next_state = np.array([id, float(i + 1)])

            log_prob = dist.log_prob(action).unsqueeze(0)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=cuda))
            masks.append(torch.tensor([1-done], dtype=torch.float, device=cuda))
            state = next_state


            if i == 99:
                # if iter == 0 or iter == 99:
                print('Iteration: {}, Score: {}'.format(iter + 1, sum(tot_reward)))
                r_graph.append(sum(tot_reward))
                break
        next_state = torch.FloatTensor(next_state).to(cuda)
        next_value = critic(next_state)
        ret_val = return_calc(next_value, rewards, masks)

        log_probs = torch.cat(log_probs)
        ret_val = torch.cat(ret_val).detach()
        values = torch.cat(values)
        advantage = ret_val - values
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        actor_optimizer.step()
        critic_optimizer.step()
    torch.save(actor, 'model/actor.pkl')
    torch.save(critic, 'model/critic.pkl')
    return r_graph


if __name__ == '__main__':
    state_size = 2
    action_size = 2
    lr = 0.0001
    print('Device: ')
    print(cuda)
    if os.path.exists('model/actor.pkl'):
        actor = torch.load('model/actor.pkl')
        print('Actor Model loaded')
    else:
        actor = Actor(state_size, action_size).to(cuda)
    if os.path.exists('model/critic.pkl'):
        critic = torch.load('model/critic.pkl')
        print('Critic Model loaded')
    else:
        critic = Critic(state_size, action_size).to(cuda)

    j2_norm = pd.read_csv('jester_2_norm.csv')
    start = time.time()
    # for i in range(0, 5):
    id = 10.0
    user_ratings = list(j2_norm.iloc[int(id)])
    r_graph = trainIters(id, user_ratings, actor, critic, tot_iterations = 100)
    elapsed = time.time() - start
    print(elapsed)
    x = list(range(1,101))
    y = r_graph
    plt.plot(x, y, color='g')
    plt.xlabel('Iterations')
    plt.ylabel('Recommender Score')
    plt.title('Iterations vs Score')
    plt.show()

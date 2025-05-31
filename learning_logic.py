import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque

class QLearning:
    ''' Class governing the Q-learning logic'''
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.Q = {}
        self.actions = actions
        self.alpha = 0.1 # learning reate
        self.gamma = 0.9 # discount factor
        self.epsilon = 0.2 # exploration rate

    @staticmethod
    def get_state(layout, pos, goal_pos, local_view_size=3):
        dx = (goal_pos[0] - pos[0]) // layout.step
        dy = (goal_pos[1] - pos[1]) // layout.step
        return (layout.local_view((pos[0]//layout.step,pos[1]//layout.step), local_view_size), dx, dy)

    def get_action(self, state):
        if state not in self.Q:
            self.Q[state] = {action: 0.0 for action in self.actions}
        
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return max(self.Q[state], key=self.Q[state].get)

    def update(self, state, action, reward, next_state):
        if next_state not in self.Q:
            self.Q[next_state] = {action: 0.0 for action in self.actions}
        max_future = max(self.Q[next_state].values())
        self.Q[state][action] += self.alpha * (reward + self.gamma * max_future - self.Q[state][action])

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
    
class DQNAgent:
    def __init__(self, state_size, action_size, device="cpu"):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        self.gamma = 0.99
        self.epsilon = 0.2
        self.batch_size = 64
        self.lr = 1e-3
        
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.replay_buffer = ReplayBuffer(10000)
        self.update_target_steps = 1000
        self.step_count = 0
    
    @staticmethod
    def get_state(layout, agent_pos, goal_pos, local_view_size=3):
        view = layout.local_view((agent_pos[0]//layout.step,agent_pos[1]//layout.step), local_view_size)
        view_flat = np.array(view).flatten()
        view_flat = [int(x == '.') for x in view_flat[0]]
        goal_dx = (goal_pos[0] - agent_pos[0]) / layout.width
        goal_dy = (goal_pos[1] - agent_pos[1]) / layout.height
        return np.concatenate((view_flat, [goal_dx, goal_dy]))
    
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return torch.argmax(q_values).item()
    
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = self.replay_buffer.sample(self.batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        reward = np.clip(reward, -1.0, 1.0)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        
        q_values = self.policy_net(state).gather(1, action)
        next_q = self.target_net(next_state).max(1)[0].unsqueeze(1)
        expected_q = reward + self.gamma * next_q.detach() * (1 - done.float())
        
        loss = F.smooth_l1_loss(q_values, expected_q)
        # print("current loss:", loss.item())
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.step_count += 1
        if self.step_count % self.update_target_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)
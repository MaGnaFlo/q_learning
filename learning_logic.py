import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from collections import deque
from astar import AStar
from game import Game

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

class ExpertDataset(Dataset):
    ''' Responsible for storing pre-computed (state, action) values for behavior cloning '''
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        state, action = self.data[idx]
        return torch.FloatTensor(state), torch.LongTensor([action])

class AStarDQN:
    def __init__(self, input_dim, num_actions, game: Game):
        self.model = DQN(input_dim, num_actions)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()
        self.game = game

    def run_expert(self, n_epochs=1000):
        max_iters = 1000
        dataset = []
        for epoch in range(n_epochs):
            self.game.generate()

            for _ in range(max_iters):
                path = AStar.search(self.game.layout, self.game.agent.pos, self.game.goals[0].pos)
                if len(path) > 1:
                    if path[1] == self.game.goals[0].pos:
                        x, y = path[1]
                        xa, ya = self.game.agent.pos
                        xg, yg = self.game.goals[0].pos
                        
                        # gather the state
                        view = self.game.layout.local_view((xa//self.game.step, ya//self.game.step),
                                                            self.game.local_view_range)
                        view_flat = np.array(view).flatten()
                        view_flat = [int(x == '.') for x in view_flat[0]]
                        goal_dx = (xg - xa) / self.game.width
                        goal_dy = (yg - ya) / self.game.height
                        
                        # find the action
                        if xa - x < 0:
                            action = 0
                        elif xa - x > 0:
                            action = 2
                        elif ya - y < 0:
                            action = 3
                        else:
                            action = 1
                        
                        dataset.append((np.concatenate((view_flat, [goal_dx, goal_dy])), action))
                        
                        self.game.generate()
                        print("AStar running: epoch", epoch)
                        
                    self.game.agent.pos = path[1]
                    
                else:
                    self.game.generate()
                    break
                self.game.move_goals()

        self.dataset = ExpertDataset(dataset)
    
    def train_from_expert(self, n_epochs):
        loader = DataLoader(self.dataset, batch_size=32, shuffle=True)
        for epoch in range(n_epochs):
            for state, action in loader:
                logits = self.model(state)
                loss = self.criterion(logits, action.squeeze())
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            print(f"Epoch {epoch}, Loss: {loss.item():.8f}")
    
    def update(self, dqn_agent, mode, epoch, training=True):
        agent = self.game.agent
        goals = self.game.goals
        actions = self.game.actions
        layout = self.game.layout
        
        # use the closest goal
        goal = min([goal for goal in goals if goal.hp > 0], key=lambda g: (agent.pos[0] - g.pos[0])**2 + (agent.pos[1] - g.pos[1])**2)
        state = DQNAgent.get_state(layout, agent.pos, goal.pos, local_view_size=self.game.local_view_range)
        action_id = dqn_agent.get_action(state)
        action = actions[action_id]

        if mode == 'killer':
            strike = False
            x_strike, y_strike = agent.pos
            reset = False
            
            reward = 0
            done = False
                
            if  action.startswith('strike_'):
                strike = True
                if action == 'strike_right':
                    x_strike += layout.step
                elif action == 'strike_left':
                    x_strike -= layout.step
                elif action == 'strike_up':
                    y_strike -= layout.step
                elif action == 'strike_down':
                    y_strike += layout.step
                
                if [x_strike, y_strike] == goal.pos:
                    goal.hp -= 1
                    if goal.hp == 0:
                        reward = 10.0
                    else:
                        reward = 7.0
                    reset = True
                else:
                    reward = -5
            else:
                new_pos = self.game.move_agent(layout, agent.pos, action)
                agent.pos = new_pos
            
                if (abs(new_pos[0] - goal.pos[0]) + abs(new_pos[1] - goal.pos[1])) // layout.step == 1:
                    reward = 1
                else:
                    reward = -0.01
                    
            self.game.move_goals(layout, goals)
            next_state = DQNAgent.get_state(layout, agent.pos, goal.pos, local_view_size=self.game.local_view_range)
            
            if training:
                dqn_agent.replay_buffer.push(state, action_id, reward, next_state, done)
                dqn_agent.update()
                
            return reset, strike, x_strike, y_strike
        
    def train(self, mode, n_epochs):
        self.dqn_agent = DQNAgent(state_size=self.game.local_view_range**2+2, 
                             action_size=len(self.game.actions), device='cpu')
        self.dqn_agent.policy_net.load_state_dict(self.model.state_dict())
        self.dqn_agent.target_net.load_state_dict(self.model.state_dict())
        
        iter = 0
        iters = []
        epoch = 0
        self.game.generate()
        
        while epoch < n_epochs:
            if iter > 10000:
                iter = 0
                print(self.game.layout)
                self.game.generate()
                continue
            
            iter += 1
            self.update(self.dqn_agent, mode, epoch, training=True)
            all_dead = True
            for goal in self.game.goals:
                if goal.hp == 0:
                    # print(f"EPOCH {epoch} ok after {iter} iterations")
                    pass
                else:
                    all_dead = False
                    
            # if all dead, reset the game
            if all_dead:
                if epoch > 0 and epoch % 1 == 0:
                    print(f"epoch: {epoch} | mean iters: {int(np.mean(iters[:-100]))} iterations")
                self.game.generate()
                epoch += 1
                iter = 0
        
        
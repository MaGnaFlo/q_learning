import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
from game import Game
import matplotlib.pyplot as plt

class QLearning:
    ''' Class governing the Q-learning logic'''
    def __init__(self, game: Game, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.Q = {}
        self.game = game
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
            self.Q[state] = {action: 0.0 for action in self.game.actions}
        
        if random.random() < self.epsilon:
            return random.choice(self.game.actions)
        else:
            return max(self.Q[state], key=self.Q[state].get)

    def update_q(self, state, action, reward, next_state):
        if next_state not in self.Q:
            self.Q[next_state] = {action: 0.0 for action in self.game.actions}
        max_future = max(self.Q[next_state].values())
        self.Q[state][action] += self.alpha * (reward + self.gamma * max_future - self.Q[state][action])
    
    def update(self, epoch, training=True):
        ''' Updates the agents' positions and Q '''
        layout = self.game.layout
        agent = self.game.agent
        goals = self.game.goals
        local_view_size = self.game.local_view_range
        mode = self.game.mode
        
        # use the closest goal
        goal = min([goal for goal in goals if goal.hp > 0], key=lambda g: (agent.pos[0] - g.pos[0])**2 + (agent.pos[1] - g.pos[1])**2)
        state = QLearning.get_state(layout, agent.pos, goal.pos, local_view_size=local_view_size)
        action = self.get_action(state)
        
        if mode == 'killer':
            strike = False
            x_strike, y_strike = agent.pos
            reset = False
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
                self.game.move_agent(layout, agent.pos, action)
            
                if (abs(agent.pos[0] - goal.pos[0]) + abs(agent.pos[1] - goal.pos[1])) // layout.step == 1:
                    reward = 1
                else:
                    reward = -0.01

            if epoch > 100:
                self.game.move_goals()
                
            if training:
                new_state = QLearning.get_state(layout, agent.pos, goal.pos)
                self.update_q(state, action, reward, new_state)
            return reset, strike, x_strike, y_strike
        
        elif mode == 'emissary':
            previous_dist = (agent.pos[0] - goal.pos[0])**2 + (agent.pos[1] - goal.pos[1])**2
            self.game.move_agent(layout, agent.pos, action)
            new_dist = (agent.pos[0] - goal.pos[0])**2 + (agent.pos[1] - goal.pos[1])**2
            reset = False
            
            if (abs(agent.pos[0] - goal.pos[0]) + abs(agent.pos[1] - goal.pos[1])) // layout.step <= 1:
                goal.hp -= 1
                reward = 5
                reset = True
            else:
                reward = previous_dist - new_dist
            
            if epoch > 0:
                self.game.move_goals()
                    
            if training:
                new_state = QLearning.get_state(layout, agent.pos, goal.pos)
                self.update_q(state, action, reward, new_state)
            return reset, False, -1, -1
    
    def train(self, num_epochs, show_stats=False, display_rate=1000):
        iters = []
        epochs = []
        iter = 0
        epoch = 0

        self.game.generate()
        previous_Q = dict(self.Q)
        
        while epoch < num_epochs:
            if iter > 10000:
                iter = 0
                print(self.game.layout)
                self.game.generate()
                self.Q = dict(previous_Q) # discard if trash
                continue
            
            iter += 1
            self.update(epoch)
            all_dead = True
            for goal in self.game.goals:
                if goal.hp != 0:
                    all_dead = False
                    
            # if all dead, reset the game
            if all_dead:
                if epoch > 0 and epoch % display_rate == 0:
                    print(f"epoch: {epoch} | mean iters: {int(np.mean(iters[:-display_rate]))} iterations")
                self.game.generate()
                epoch += 1
                iter = 0
                previous_Q = dict(self.Q)
                
            epochs.append(epoch)
            iters.append(iter)
        
        if show_stats:
            plt.scatter(epochs[::display_rate], iters[::display_rate], s=1)
            plt.show()

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
    
class DQNLearning:
    def __init__(self, game: Game, state_size, action_size, device="cpu"):
        self.game = game
        
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        self.gamma = 0.99
        self.batch_size = 64
        self.lr = 1e-3
        
        self.epsilon_start = 1.0
        self.epsilon_final = 0.05
        self.epsilon_decay = 10000
        
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.replay_buffer = ReplayBuffer(10000)
        self.update_target_steps = 1000
        self.step_count = 0
        self.last_reward = 0# keep track of rewards per epoch
    
    def get_state(self, goal_pos):
        layout = self.game.layout
        agent_pos = self.game.agent.pos
        
        view = layout.local_view((agent_pos[0]//layout.step,agent_pos[1]//layout.step), self.game.local_view_range)
        view_flat = np.array(view).flatten()
        view_flat = [int(x == '.') for x in view_flat[0]]
        goal_dx = (goal_pos[0] - agent_pos[0]) / layout.width
        goal_dy = (goal_pos[1] - agent_pos[1]) / layout.height
        
        return np.concatenate((view_flat, [goal_dx, goal_dy]))
    
    def get_action(self, state):
        epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                    np.exp(-self.step_count / self.epsilon_decay)
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return torch.argmax(q_values).item()
    
    def update_(self):
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
    
    def update(self, training=True):
        agent = self.game.agent
        layout = self.game.layout
        goals = self.game.goals
        local_view_size = self.game.local_view_range
        actions = self.game.actions
        mode = self.game.mode
        
        ''' Updates the agents' positions and Q '''
        # use the closest goal
        goal = min([goal for goal in goals if goal.hp > 0], key=lambda g: (agent.pos[0] - g.pos[0])**2 + (agent.pos[1] - g.pos[1])**2)
        state = self.get_state(goal.pos)
        action_id = self.get_action(state)
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
                    if (agent.pos[0] - goal.pos[0])**2 + (agent.pos[1] - goal.pos[1])**2 > 5:
                        reward = -10
            else:
                previous_dist = (abs(agent.pos[0] - goal.pos[0]) + abs(agent.pos[1] - goal.pos[1])) // layout.step
                self.game.move_agent(layout, agent.pos, action)
                new_dist = (agent.pos[0] - goal.pos[0])**2 + (agent.pos[1] - goal.pos[1])**2
            
                if new_dist < 5:
                    reward = -5
                else:
                    reward = (previous_dist - new_dist) / self.game.width / self.game.height
                    
            self.game.move_goals()
            next_state = self.get_state(goal.pos)
            
            if training:
                self.replay_buffer.push(state, action_id, reward, next_state, done)
                self.update_()
            
            self.last_reward += reward
            return reset, strike, x_strike, y_strike
        
        elif mode == 'emissary':
            previous_dist = (agent.pos[0] - goal.pos[0])**2 + (agent.pos[1] - goal.pos[1])**2
            self.game.move_agent(layout, agent.pos, action)
            new_dist = (agent.pos[0] - goal.pos[0])**2 + (agent.pos[1] - goal.pos[1])**2
            reset = False
                
            reward = 0
            done = False
                
            if (abs(agent.pos[0] - goal.pos[0]) + abs(agent.pos[1] - goal.pos[1])) // layout.step<= 1:
                goal.hp -= 1
                reward = 2
                reset = True
            else:
                reward = (previous_dist - new_dist) / self.game.width / self.game.height
                
            self.game.move_goals()
            next_state = self.get_state(goal.pos)
                
            if training:
                self.replay_buffer.push(state, action_id, reward, next_state, done)
                self.update_()
            
            self.last_reward += reward
            return reset, False, -1, -1
    
    def train(self, num_epochs, show_stats=False, display_rate=10):
        ''' Gotta go fast '''
        iters = []
        rewards = []
        epochs = []
        iter = 0
        epoch = 0
        self.game.generate()
        
        while epoch < num_epochs:
            if iter > 10000:
                iter = 0
                print(self.game.layout)
                self.game.generate()
                continue
            
            iter += 1
            self.update()
            all_dead = True
            for goal in self.game.goals:
                if goal.hp != 0:
                    all_dead = False
                    break
                    
            # if all dead, reset the game
            if all_dead:
                epochs.append(epoch)
                iters.append(iter)
                rewards.append(self.last_reward / iter)
                self.last_reward = 0
                
                if epoch > 0 and epoch % display_rate == 0:
                    print(f"epoch {epoch} | {int(np.mean(iters[:-display_rate]))} iterations | {int(np.mean(rewards[:-display_rate]))} rewards")
                self.game.generate()
            
                epoch += 1
                iter = 0
        
        if show_stats:
            _, ax = plt.subplots(1, 2, figsize=(10,5))
            ax[0].scatter(epochs[::display_rate], iters[::display_rate], s=2, c='teal')
            ax[1].scatter(epochs[::display_rate], rewards[::display_rate], s=2, c='orange')
            plt.show()
        
        return epoch
        
    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)
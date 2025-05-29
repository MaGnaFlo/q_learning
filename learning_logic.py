import random

class QLearning:
    ''' Class governing the Q-learning logic'''
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.Q = {}
        self.actions = actions
        self.alpha = 0.1 # learning reate
        self.gamma = 0.9 # discount factor
        self.epsilon = 0.2 # exploration rate

    @staticmethod
    def get_state(layout, pos, goal_pos):
        dx = (goal_pos[0] - pos[0]) // layout.step
        dy = (goal_pos[1] - pos[1]) // layout.step
        return (layout.local_view((pos[0]//layout.step,pos[1]//layout.step), 7), dx, dy)

    def choose_action(self, state):
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
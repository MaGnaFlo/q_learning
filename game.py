from layout import Layout
from agent import Agent
import random

class Game:
    def __init__(self, 
                 width, height, wall_density, step, local_view_range,
                 n_goals, hp,
                 actions):
        self.width = width
        self.height = height
        self.step = step
        self.density = wall_density
        self.local_view_range = local_view_range
        self.n_goals = n_goals
        self.hp = hp
        self.actions = actions
    
    def generate(self):
        self.layout = Layout(self.width, self.height, step=self.step)
        self.layout.generate(d=self.density)
        
        self.agent = Agent(self.layout.random_position(), self.layout.step, self.layout.step, (0,255,200))
        self.agent.pos = self.layout.random_position()
        self.goals = []
        for _ in range(self.n_goals):
            goal = Agent(self.layout.random_position(), self.layout.step, self.layout.step,
                        (random.randint(0,255),random.randint(0,255),random.randint(0,255)), 
                        hp=self.hp)
            self.goals.append(goal)
            
    def move_agent(self, layout, pos, action):
        ''' Finds the next position according to an action and a layout '''
        x, y = pos
        if action == 'right':
            x += layout.step
        elif action == 'left':
            x -= layout.step
        elif action == 'down':
            y += layout.step
        elif action == 'up':
            y -= layout.step
        x = max(0, min(x, layout.step * (layout.width-1)))
        y = max(0, min(y, layout.step * (layout.height-1)))
        if layout.passable((x,y)):
            self.agent.pos = [x,y]
            
    def move_goals(self):
        ''' Randomly moves the goals whenever possible in the layout'''
        for goal in self.goals:
            x, y = goal.pos
            r = random.randint(0,3)
            if r == 0:
                x += self.layout.step
            elif r == 1:
                x -= self.layout.step
            elif r == 2:
                y += self.layout.step
            else:
                y -= self.layout.step
            x = max(0, min(x, self.layout.step * (self.layout.width-1)))
            y = max(0, min(y, self.layout.step * (self.layout.height-1)))
            if self.layout.passable((x,y)):
                goal.pos = [x,y]
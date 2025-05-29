import random
from learning_logic import QLearning

class Agent:
    ''' Actor of the game '''
    def __init__(self, pos, width, height, color, hp=1):
        self.pos = pos
        self.width = width
        self.height = height
        self.color = color
        self.hp = hp
    
    def rect(self):
        return [self.pos[0], self.pos[1], self.width, self.height]

    @staticmethod
    def move(layout, pos, action):
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
        if not layout.passable((x,y)):
            return pos
        return [x,y]

    @staticmethod
    def move_goals(layout, goals):
        ''' Randomly moves the goals whenever possible in the layout'''
        for goal in goals:
            x, y = goal.pos
            r = random.randint(0,3)
            if r == 0:
                x += layout.step
            elif r == 1:
                x -= layout.step
            elif r == 2:
                y += layout.step
            else:
                y -= layout.step
            x = max(0, min(x, layout.step * (layout.width-1)))
            y = max(0, min(y, layout.step * (layout.height-1)))
            if layout.passable((x,y)):
                goal.pos = [x,y]

    @staticmethod
    def update(Q, layout, agent, goals, epoch):
        ''' Updates the agents' positions and Q '''
        # use the closest goal
        goal = min([goal for goal in goals if goal.hp > 0], key=lambda g: abs(agent.pos[0] - g.pos[0]) + abs(agent.pos[1] - g.pos[1]))
        state = QLearning.get_state(layout, agent.pos, goal.pos)
        action = Q.choose_action(state)
        
        strike = False
        x_strike, y_strike = agent.pos
        reset = False
        if action.startswith('strike_'):
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
            new_pos = Agent.move(layout, agent.pos, action)
            agent.pos = new_pos
            reset = False
        
            if (abs(new_pos[0] - goal.pos[0]) + abs(new_pos[1] - goal.pos[1])) // layout.step <= 1:
                reward = 1
            else:
                reward = -0.01
            
        new_state = QLearning.get_state(layout, agent.pos, goal.pos)
        
        if epoch > 10:
            Agent.move_goals(layout, goals)
        
        Q.update(state, action, reward, new_state)
        return reset, strike, x_strike, y_strike
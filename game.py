import random

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

class Layout:
    ''' This is the layout of the map '''
    def __init__(self, width, height, step=32):
        self.width = width
        self.height = height
        self.grid = []
        self.step = step
    
    def __repr__(self):
        return "\n".join("".join(row) for row in self.grid)
    
    def generate(self, d=0.5):
        self.grid = [['#' for _ in range(self.width)] for _ in range(self.height)]
        
        x, y = random.randint(1, self.width-2), random.randint(1, self.height-1)
        
        n = (self.width-2)*(self.height-2)
        n_current = n
        while n_current > int(d * n):
            r = random.randint(0,3)
            x += 1 if r == 0 else -1 if r == 1 else 0
            y += 1 if r == 2 else -1 if r == 3 else 0
            
            x = max(1, min(self.width - 2, x))
            y = max(1, min(self.height - 2, y))
            
            if self.grid[y][x] == '#':
                self.grid[y][x] = '.'
                n_current -= 1

    def local_view(self, pos, n=3):
        ''' Returns a state representation of the position's surroundings '''
        ax, ay = pos
        view = ""
        for i in range(-(n//2), n//2+1):
            row = []
            x = ax + i
            for j in range(-(n//2), n//2+1):
                y = ay + j
                if 0 <= x < self.width and 0 <= y < self.height:
                    row.append(self.grid[y][x])
                else:
                    row.append('#')
            view += "".join(row)
        return "".join(view)

    def passable(self, pos):
        x, y = pos
        return self.grid[y//self.step][x//self.step] != '#'

    def to_grid(self, pos):
        return [pos[0] // self.step, pos[1] // self.step]

    def to_real(self, grid_pos):
        return [grid_pos[0] * self.step, grid_pos[1] * self.step]

    def random_position(self):
        ok = False
        while not ok:
            x, y = self.step * random.randint(0, self.width-1), self.step * random.randint(0, self.height-1)
            ok = self.passable((x,y))
        return [x,y]
    
class Game:
    def __init__(self, 
                 width, height, wall_density, step, local_view_range,
                 n_goals, hp,
                 actions,
                 mode):
        self.width = width
        self.height = height
        self.step = step
        self.density = wall_density
        self.local_view_range = local_view_range
        self.n_goals = n_goals
        self.hp = hp
        self.actions = actions
        self.mode = mode
    
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
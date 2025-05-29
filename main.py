import pygame, sys
from pygame.locals import *
import random

import matplotlib.pyplot as plt

pygame.init()

BACKGROUND = (0,0,0)
FPS = 4
fpsClock = pygame.time.Clock()
WIDTH = 640
HEIGHT = 480

STEP = 32

MAP = [
    "####################",
    "#......#...........#",
    "#..............###.#",
    "#............#...#.#",
    "###........##...#..#",
    "#.....#.......#....#",
    "#.###.#.......#..###",
    "#..................#",
    "#.#.###.......###..#",
    "#.#.....#.#.#...#..#",
    "#.........#.....#..#",
    "#.........#......#.#",
    "###...####.........#",
    "#.....#............#",
    "####################",
]

MAP_HEIGHT = len(MAP)
MAP_WIDTH = len(MAP[0])

def crop(val, axis):
    if axis == 'x':
        return max(0, min(val, MAP_WIDTH-1))
    elif axis == 'y':
        return max(0, min(val, MAP_HEIGHT-1))

def local_view(pos, radius=3):
    ax, ay = pos
    view = (MAP[crop(ay-radius//2, 'y')]    [crop(ax-radius//2, 'x'):crop(ax+radius//2+1, 'x')],
            MAP[crop(ay, 'y')]              [crop(ax-radius//2, 'x'):crop(ax+radius//2+1, 'x')],
            MAP[crop(ay+radius//2, 'y')]    [crop(ax-radius//2, 'x'):crop(ax+radius//2+1, 'x')]
    )
    return "".join(view)

def display_map(window):
    for y in range(0, MAP_HEIGHT):
        for x in range(0, MAP_WIDTH):
            if MAP[y][x] == '#':
                pygame.draw.rect(window, (100,100,100),
                                 (STEP*x,STEP*y,STEP,STEP))
def passable(pos):
    x, y = pos
    return MAP[y//STEP][x//STEP] != '#'

def place():
    ok = False
    while not ok:
        x, y = STEP * random.randint(0, MAP_WIDTH-1), STEP * random.randint(0, MAP_HEIGHT-1)
        ok = passable((x,y))
    return [x,y]

window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Q-learning")

Q = {}
ACTIONS = ['right', 'up', 'left', 'down',
           'strike_right', 'strike_up', 'strike_left', 'strike_down']
ALPHA = 0.1 # learning reate
GAMMA = 0.9 # discount factor
EPSILON = 0.2 # exploration rate

EPOCH = 0
ITER = 0

class Agent:
    def __init__(self, x, y, color, hp=1):
        self.pos = [x,y]
        self.width = STEP
        self.height = STEP
        self.color = color
        self.hp = hp
    
    def rect(self):
        return [self.pos[0], self.pos[1], self.width, self.height]

def get_state(pos, goal_pos):
    dx = (goal_pos[0] - pos[0]) // STEP
    dy = (goal_pos[1] - pos[1]) // STEP
    return (local_view((pos[0]//STEP,pos[1]//STEP), 7), dx, dy)

def choose_action(state):
    if state not in Q:
        Q[state] = {action: 0.0 for action in ACTIONS}
    
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    else:
        return max(Q[state], key=Q[state].get)

def update_Q(state, action, reward, next_state):
    if next_state not in Q:
        Q[next_state] = {action: 0.0 for action in ACTIONS}
    max_future = max(Q[next_state].values())
    Q[state][action] += ALPHA * (reward + GAMMA * max_future - Q[state][action])

def move(pos, action):
    x, y = pos
    if action == 'right':
        x += STEP
    elif action == 'left':
        x -= STEP
    elif action == 'down':
        y += STEP
    elif action == 'up':
        y -= STEP
    x = max(0, min(x, WIDTH - STEP))
    y = max(0, min(y, HEIGHT - STEP))
    if not passable((x,y)):
        return pos
    return [x,y]

def move_goal(goal):
    x, y = goal.pos
    r = random.randint(0,3)
    if r == 0:
        x += STEP
    elif r == 1:
        x -= STEP
    elif r == 2:
        y += STEP
    else:
        y -= STEP
    x = max(0, min(x, WIDTH - STEP))
    y = max(0, min(y, HEIGHT - STEP))
    if passable((x,y)):
        goal.pos = [x,y]
    
    
def update(agent, goal):
    global EPOCH, ITER
    state = get_state(agent.pos, goal.pos)
    action = choose_action(state)
    
    strike = False
    x_strike, y_strike = agent.pos
    reset = False
    if action.startswith('strike_'):
        strike = True
        if action == 'strike_right':
            x_strike += STEP
        elif action == 'strike_left':
            x_strike -= STEP
        elif action == 'strike_up':
            y_strike -= STEP
        elif action == 'strike_down':
            y_strike += STEP
        
        if [x_strike, y_strike] == goal.pos:
            reward = 10.0
            reset = True
        else:
            reward = -0.05
    else:
        new_pos = move(agent.pos, action)
        agent.pos = new_pos
        reset = False
    
        if (abs(new_pos[0] - goal.pos[0]) + abs(new_pos[1] - goal.pos[1])) // STEP <= 1:
            reward = 1
        else:
            reward = -0.01

        
    new_state = get_state(agent.pos, goal.pos)
    
    if EPOCH > 10:
        move_goal(goal)
    
    update_Q(state, action, reward, new_state)
    return reset, strike, x_strike, y_strike

def main():
    global EPOCH, ITER
    EPOCH = 0
    ITER = 0
    
    reset_pending = False
    reset_counter = 0
    
    agent = Agent(32,32, (0,255,200))
    goal = Agent(32*11, 32*10, (255,255,0))
    
    iters = []
    epochs = []
    
    # fast training
    while EPOCH < 100000:
        ITER += 1
        reset, _, _, _ = update(agent, goal)
        if reset:
            agent.pos = place()
            goal.pos = place()
            print(f"EPOCH {EPOCH} ok after {ITER} iterations")
            EPOCH += 1
            ITER = 0

        epochs.append(EPOCH)
        iters.append(ITER)
    
    # plt.scatter(epochs[::1000], iters[::1000], s=1)
    # plt.show()
    
    # display result
    loop = True
    while loop:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        ITER += 1
        
        
        window.fill(BACKGROUND)
        display_map(window)
        
        pygame.draw.rect(window, goal.color, goal.rect())
        pygame.draw.rect(window, agent.color, agent.rect())
        
        if reset_pending:
            reset_counter += 1
            if reset_counter >= 1: # one frame delay
                agent.pos = place()
                goal.pos = place()
                print(f"EPOCH {EPOCH} ok after {ITER} iterations")
                EPOCH += 1
                ITER = 0
                reset_counter = 0
                reset_pending = False
        else:
            reset, strike, x_strike, y_strike = update(agent, goal)
            if strike:
                pygame.draw.circle(window, (255,0,0), (x_strike+STEP//2, y_strike+STEP//2), 16)
            if reset:
                reset_pending = True
        
        pygame.display.update()
        fpsClock.tick(FPS)
            
        

main()
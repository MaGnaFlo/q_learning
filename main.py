from learning_logic import QLearning
from layout import Layout
from agent import Agent
from screen import Screen

import sys
import random
import matplotlib.pyplot as plt

N_GOALS = 5
HP = 3
ACTIONS = ['right', 'up', 'left', 'down',
            'strike_right', 'strike_up', 'strike_left', 'strike_down']

def fast_train(Q, layout, agent, goals, max_epochs=100000, show_stats=False):
    ''' Gotta go fast '''
    iters = []
    epochs = []
    iter = 0
    epoch = 0
    while epoch < max_epochs:
        iter += 1
        Agent.update(Q, layout, agent, goals, epoch)
        for goal in goals:
            if goal.hp == 0:
                goal.pos = layout.random_position()
                goal.hp = HP
                print(f"EPOCH {epoch} ok after {iter} iterations")
                epoch += 1
                iter = 0
                break

        epochs.append(epoch)
        iters.append(iter)
    
    if show_stats:
        plt.scatter(epochs[::1000], iters[::1000], s=1)
        plt.show()
    
    return epoch

##########################################################################

def main():
    fast_train_max_epochs = 100000
    if len(sys.argv) > 1:
        fast_train_max_epochs = int(sys.argv[1])
    
    layout = Layout(25, 19)
    layout.generate(scarcity=0.9)
    
    # this is just to let the screen show the last action before reset
    reset_pending = False
    reset_counter = 0
    
    # create agent and goals
    agent = Agent(layout.random_position(), layout.step, layout.step, (0,255,200))
    agent.pos = layout.random_position()
    goals = []
    for _ in range(N_GOALS):
        goal = Agent(layout.random_position(), layout.step, layout.step,
                     (random.randint(0,255),random.randint(0,255),random.randint(0,255)), 
                     hp=HP)
        goals.append(goal)
        
    # fast training
    Q = QLearning(ACTIONS)
    epoch = fast_train(Q, layout, agent, goals, max_epochs=fast_train_max_epochs, show_stats=False)
    
    # display result
    screen = Screen(layout, fps=8)
    loop = True
    iter = 0
    while loop:
        Screen.register_events()
        iter += 1
        
        screen.draw_background()
        screen.draw_layout()
        
        # draw agents
        for goal in goals:
            if goal.hp > 0:
                screen.draw_agent(goal.color, goal.pos)
        screen.draw_agent(agent.color, agent.pos)
        
        if reset_pending:
            reset_counter += 1
            if reset_counter >= 1: # one frame delay
                for goal in goals:
                    if goal.hp == 0:
                        goal.pos = layout.random_position()
                        goal.hp = HP
                        print(f"EPOCH {epoch} ok after {iter} iterations")
                        epoch += 1
                        iter = 0
                reset_counter = 0
                reset_pending = False
        else:
            reset, strike, x_strike, y_strike = Agent.update(Q, layout, agent, goals, epoch)
            if strike:
                screen.draw_strike((x_strike, y_strike))
            if reset:
                reset_pending = True
        
        screen.update()
        
if __name__ == '__main__':
    main()
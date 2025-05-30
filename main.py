from learning_logic import QLearning
from layout import Layout
from agent import Agent
from screen import Screen
import numpy as np
import sys
import random
import matplotlib.pyplot as plt

W, H = 19, 19
N_GOALS = 1
HP = 1
ACTIONS = {
    'emissary': ['right', 'up', 'left', 'down'],
    'killer': ['right', 'up', 'left', 'down',
               'strike_right', 'strike_up', 'strike_left', 'strike_down']
}
LAYOUT_SCARCITY = 1

def fast_train(Q, mode, max_epochs=100000, show_stats=False):
    ''' Gotta go fast '''
    iters = []
    epochs = []
    iter = 0
    epoch = 0
    layout, agent, goals = set_game()
    previous_Q = Q.Q
    
    while epoch < max_epochs:
        if iter > 10000:
            iter = 0
            print(layout)
            layout, agent, goals = set_game()
            Q.Q = previous_Q # discard if trash
            continue
        
        iter += 1
        Agent.update(Q, mode, layout, agent, goals, epoch)
        all_dead = True
        for goal in goals:
            if goal.hp == 0:
                # print(f"EPOCH {epoch} ok after {iter} iterations")
                pass
            else:
                all_dead = False
                
        # if all dead, reset the game
        if all_dead:
            if epoch > 0 and epoch % 1000 == 0:
                print(f"epoch: {epoch} | mean iters: {int(np.mean(iters[:-1000]))} iterations")
            layout, agent, goals = set_game()
            epoch += 1
            iter = 0
            previous_Q = Q.Q
            
        epochs.append(epoch)
        iters.append(iter)
    
    if show_stats:
        plt.scatter(epochs[::1000], iters[::1000], s=1)
        plt.show()
    
    return epoch

def set_game():
    # layout
    layout = Layout(W, H)
    layout.generate(scarcity=LAYOUT_SCARCITY)
    
    # create agent and goals
    agent = Agent(layout.random_position(), layout.step, layout.step, (0,255,200))
    agent.pos = layout.random_position()
    goals = []
    for _ in range(N_GOALS):
        goal = Agent(layout.random_position(), layout.step, layout.step,
                     (random.randint(0,255),random.randint(0,255),random.randint(0,255)), 
                     hp=HP)
        goals.append(goal)
    return layout, agent, goals

##########################################################################

def main():
    fast_train_max_epochs = 100000
    mode = 'killer'
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    if len(sys.argv) > 2:
        fast_train_max_epochs = int(sys.argv[2])
    
    # this is just to let the screen show the last action before reset
    reset_pending = False
    reset_counter = 0

    # fast training
    Q = QLearning(ACTIONS[mode], epsilon=0.05)
    epoch = fast_train(Q, mode, max_epochs=fast_train_max_epochs, show_stats=True)
    
    # display result
    Q.epsilon = 0.05
    layout, agent, goals = set_game()
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
                all_dead = True
                for goal in goals:
                    if goal.hp == 0:
                        print(f"EPOCH {epoch} ok after {iter} iterations")
                        epoch += 1
                        iter = 0
                    else:
                        all_dead = False
                        
                # if there isn't any goal left, reset the game with a different layout
                if all_dead:
                    layout, agent, goals = set_game()
                    screen.set_layout(layout)

                reset_counter = 0
                reset_pending = False
        else:
            reset, strike, x_strike, y_strike = Agent.update(Q, mode, layout, agent, goals, epoch, training=False)
            if strike:
                screen.draw_strike((x_strike, y_strike))
            if reset:
                reset_pending = True
        
        screen.update()
        
if __name__ == '__main__':
    main()
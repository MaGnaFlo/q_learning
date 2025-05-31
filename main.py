from learning_logic import QLearning
from layout import Layout
from agent import Agent
from screen import Screen
import numpy as np
import sys
import random
import matplotlib.pyplot as plt
from learning_logic import DQNAgent
from astar import AStar

W, H = 30, 30
N_GOALS = 1
HP = 2
ACTIONS = {
    'emissary': ['none', 'right', 'up', 'left', 'down'],
    'killer': ['none', 'right', 'up', 'left', 'down', 'none'
               'strike_right', 'strike_up', 'strike_left', 'strike_down']
}
WALL_DENSITY = 0.5
LOCAL_VIEW_SIZE = 5

USE_DEEP = True

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

def fast_train_deep(dqn_agent, mode, max_epochs=100000, show_stats=False):
    ''' Gotta go fast '''
    iters = []
    epochs = []
    iter = 0
    epoch = 0
    layout, agent, goals = set_game()
    
    while epoch < max_epochs:
        if iter > 10000:
            iter = 0
            print(layout)
            layout, agent, goals = set_game()
            continue
        
        iter += 1
        Agent.update_deep(dqn_agent, ACTIONS[mode], mode, layout, agent, goals, epoch, local_view_size=LOCAL_VIEW_SIZE)
        all_dead = True
        for goal in goals:
            if goal.hp == 0:
                # print(f"EPOCH {epoch} ok after {iter} iterations")
                pass
            else:
                all_dead = False
                
        # if all dead, reset the game
        if all_dead:
            if epoch > 0 and epoch % 1 == 0:
                print(f"epoch: {epoch} | mean iters: {int(np.mean(iters[:-100]))} iterations")
            layout, agent, goals = set_game()
            epoch += 1
            iter = 0
            
        epochs.append(epoch)
        iters.append(iter)
    
    if show_stats:
        plt.scatter(epochs[::1000], iters[::1000], s=1)
        plt.show()
    
    return epoch

def set_game():
    # layout
    layout = Layout(W, H)
    layout.generate(d=WALL_DENSITY)
    
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
    if USE_DEEP:
        dqn_agent = DQNAgent(state_size=LOCAL_VIEW_SIZE**2+2, action_size=len(ACTIONS[mode]), device='cpu')
        epoch = fast_train_deep(dqn_agent, mode, max_epochs=fast_train_max_epochs, show_stats=True)
        dqn_agent.save("test.pth")
    else:
        Q = QLearning(ACTIONS[mode], epsilon=0.05)
        epoch = fast_train(Q, mode, max_epochs=fast_train_max_epochs, show_stats=True)
    
    
    # display result
    dqn_agent.epsilon = 0.0
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
            if USE_DEEP:
                reset, strike, x_strike, y_strike = Agent.update_deep(dqn_agent, ACTIONS[mode], mode, layout, agent, goals, epoch, 
                                                                  local_view_size=LOCAL_VIEW_SIZE, training=False)
            else:
                reset, strike, x_strike, y_strike = Agent.update(Q, mode, layout, agent, goals, epoch, 
                                                                 local_view_size=LOCAL_VIEW_SIZE, training=False)
            
            if strike:
                screen.draw_strike((x_strike, y_strike))
            if reset:
                reset_pending = True
        
        screen.update()

# ASTAR ####################################
def test_astar():
    layout, agent, goals = set_game()
    screen = Screen(layout, fps=8)
    loop = True
    iter = 0
    while loop:
        Screen.register_events()
        iter += 1
        
        screen.draw_background()
        screen.draw_layout()
        
        Agent.move_goals(layout, goals)
        
        path = AStar.search(layout, agent.pos, goals[0].pos)
        if len(path) > 1:
            agent.pos = path[1]
            if agent.pos == goals[0].pos:
                layout, agent, goals = set_game()
                screen = Screen(layout, fps=8)
        else:
            layout, agent, goals = set_game()
            screen = Screen(layout, fps=8)
            

        # draw agents
        for goal in goals:
            if goal.hp > 0:
                screen.draw_agent(goal.color, goal.pos)
        screen.draw_agent(agent.color, agent.pos)
        
        screen.update()
        
if __name__ == '__main__':
    # main()
    test_astar()
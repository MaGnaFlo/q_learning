from learning_logic import QLearning, DQNLearning
from astar import AStar
from screen import Screen
import sys
from game import Game

W, H = 10, 10
N_GOALS = 3
HP = 2
ACTIONS = {
    'emissary': ['right', 'up', 'left', 'down'],
    'killer': ['right', 'up', 'left', 'down',
               'strike_right', 'strike_up', 'strike_left', 'strike_down']
}
WALL_DENSITY = 0.25
LOCAL_VIEW_SIZE = 5

def test_qlearning(mode, training_epochs):
    # this is just to let the screen show the last action before reset
    reset_pending = False
    reset_counter = 0
    
    game = Game(W, H, WALL_DENSITY, 32, LOCAL_VIEW_SIZE, N_GOALS, HP, ACTIONS[mode], mode)
    game.generate()

    Q = QLearning(game)
    Q.train(num_epochs=training_epochs, display_rate=100, show_stats=False)
    
    # display result
    game.generate()
    screen = Screen(game.layout, fps=8)
    loop = True
    while loop:
        Screen.register_events()
        
        screen.draw_background()
        screen.draw_layout()
        
        # draw agents
        for goal in game.goals:
            if goal.hp > 0:
                screen.draw_agent(goal.color, goal.pos)
        screen.draw_agent(game.agent.color, game.agent.pos)
        
        if reset_pending:
            reset_counter += 1
            if reset_counter >= 1: # one frame delay
                all_dead = True
                for goal in game.goals:
                    if goal.hp != 0:
                        all_dead = False
                        
                # if there isn't any goal left, reset the game with a different layout
                if all_dead:
                    game.generate()
                    screen.set_layout(game.layout)

                reset_counter = 0
                reset_pending = False
        else:
            reset, strike, x_strike, y_strike = Q.update(101, training=False)
            if strike:
                screen.draw_strike((x_strike, y_strike))
            if reset:
                reset_pending = True
        
        screen.update()

def test_dqn(mode, training_epochs):
    # this is just to let the screen show the last action before reset
    reset_pending = False
    reset_counter = 0
    
    game = Game(W, H, WALL_DENSITY, 32, LOCAL_VIEW_SIZE, N_GOALS, HP, ACTIONS[mode], mode)
    game.generate()

    # fast training
    dqn = DQNLearning(game, state_size=LOCAL_VIEW_SIZE**2+2, action_size=len(ACTIONS[mode]), device='cpu')
    dqn.train(num_epochs=training_epochs, display_rate=1, show_stats=True)
    dqn.epsilon = 0.0
        
    # display result
    game.generate()
    screen = Screen(game.layout, fps=8)
    loop = True
    while loop:
        Screen.register_events()
        
        screen.draw_background()
        screen.draw_layout()
        
        # draw agents
        for goal in game.goals:
            if goal.hp > 0:
                screen.draw_agent(goal.color, goal.pos)
        screen.draw_agent(game.agent.color, game.agent.pos)
        
        if reset_pending:
            reset_counter += 1
            if reset_counter >= 1: # one frame delay
                all_dead = True
                for goal in game.goals:
                    if goal.hp != 0:
                        all_dead = False
                        
                # if there isn't any goal left, reset the game with a different layout
                if all_dead:
                    game.generate()
                    screen.set_layout(game.layout)

                reset_counter = 0
                reset_pending = False
        else:
            reset, strike, x_strike, y_strike = dqn.update(training=False)
            if strike:
                screen.draw_strike((x_strike, y_strike))
            if reset:
                reset_pending = True
        
        screen.update()

def test_astar():
    game = Game(W, H, WALL_DENSITY, step=32, local_view_range=LOCAL_VIEW_SIZE,
                n_goals=N_GOALS, hp=HP, actions=[], mode='')
    game.generate()
    
    screen = Screen(game.layout, fps=8)
    loop = True
    while loop:
        Screen.register_events()
        
        screen.draw_background()
        screen.draw_layout()
        
        game.move_goals()
        
        path = AStar.search(game.layout, game.agent.pos, game.goals[0].pos)
        if len(path) > 1:
            game.agent.pos = path[1]
            if game.agent.pos == game.goals[0].pos:
                game.generate()
                screen = Screen(game.layout, fps=8)
        else:
            game.generate()
            screen = Screen(game.layout, fps=8)
            

        # draw agents
        for goal in game.goals:
            if goal.hp > 0:
                screen.draw_agent(goal.color, goal.pos)
        screen.draw_agent(game.agent.color, game.agent.pos)
        
        screen.update()

def main():
    if len(sys.argv) < 4:
        print("Error: 3 arguments required (algorithm, mode, epochs)")
        
    algorithm = sys.argv[1]
    mode = sys.argv[2]
    training_epochs = int(sys.argv[3])
    
    if algorithm == 'astar':
        test_astar()
    elif algorithm == 'qlearning':
        test_qlearning(mode, training_epochs)
    elif algorithm == 'dqn':
        test_dqn(mode, training_epochs)
    else:
        print("Error: algorithm not recognized")
        
if __name__ == '__main__':
    main()
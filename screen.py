import pygame, sys
from pygame.locals import *

class Screen:
    ''' Handles the screen display with pygame'''
    def __init__(self, layout, fps=60):
        self.clock = pygame.time.Clock()
        self.fps = fps
        pygame.init()
        self.surface = pygame.display.set_mode((layout.step * layout.width, 
                                                layout.step * layout.height))
        pygame.display.set_caption("Q-learning")
        self.layout = layout
    
    def set_layout(self, layout):
        self.surface = pygame.display.set_mode((layout.step * layout.width, 
                                                layout.step * layout.height))
        self.layout = layout
        
    def draw_background(self):
        self.surface.fill((0,0,0))

    def draw_layout(self):
        for y in range(0, self.layout.height):
            for x in range(0, self.layout.width):
                if self.layout.grid[y][x] == '#':
                    pygame.draw.rect(self.surface, (100,100,100),
                                    (self.layout.step*x,self.layout.step*y,
                                     self.layout.step,self.layout.step))
    
    def draw_agent(self, color, pos):
        pygame.draw.rect(self.surface, color, 
                         (pos[0], pos[1], self.layout.step, self.layout.step))
    
    def draw_strike(self, pos):
        pygame.draw.circle(self.surface, (255,0,0), 
                           (pos[0]+self.layout.step//2, pos[1]+self.layout.step//2), 8)
    
    def update(self):
        pygame.display.update()
        self.clock.tick(self.fps)
    
    @staticmethod
    def register_events():
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()


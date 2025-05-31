import random

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

    def random_position(self):
        ok = False
        while not ok:
            x, y = self.step * random.randint(0, self.width-1), self.step * random.randint(0, self.height-1)
            ok = self.passable((x,y))
        return [x,y]

if __name__ == '__main__':
    layout = Layout(9,9)
    layout.generate2(d=0.5)
    print(layout)
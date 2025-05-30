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

    def generate(self, scarcity=0.5):
        ''' Generates a random map as a connected graph
        The scarcity parameters allows to statistically carve additional holes
        :param self
        :param scarcity: percentage of the total surface to statistically remove
        '''
        # Ensure dimensions are odd for the maze generation algorithm.
        if self.width % 2 == 0:
            self.width -= 1
        if self.height % 2 == 0:
            self.height -= 1

        # Create a grid filled with walls.
        maze = [['#' for _ in range(self.width)] for _ in range(self.height)]
        
        # Randomly choose a starting cell (should be at odd indices).
        start_x = random.randrange(1, self.width, 2)
        start_y = random.randrange(1, self.height, 2)
        maze[start_y][start_x] = '.'

        # Use a stack for the recursive backtracking.
        stack = [(start_x, start_y)]
        
        while stack:
            x, y = stack[-1]
            neighbors = []
            # Check neighbors two cells away (up, down, left, right).
            for dx, dy in [(0, 2), (2, 0), (0, -2), (-2, 0)]:
                nx, ny = x + dx, y + dy
                # Check that neighbor is inside bounds and is still a wall.
                if 0 <= nx < self.width and 0 <= ny < self.height and maze[ny][nx] == '#':
                    neighbors.append((nx, ny, dx, dy))
            if neighbors:
                # Choose a random valid neighbor.
                nx, ny, dx, dy = random.choice(neighbors)
                # Carve a passage between the current cell and the neighbor.
                maze[y + dy // 2][x + dx // 2] = '.'
                maze[ny][nx] = '.'
                stack.append((nx, ny))
            else:
                stack.pop()
        
        for _ in range(int(self.width * self.height * scarcity)):
            x = random.randrange(1, self.width - 1)
            y = random.randrange(1, self.height - 1)
            if maze[y][x] == '#':
                maze[y][x] = '.'
        
        # Return the maze as a list of strings.
        self.grid = [''.join(row) for row in maze]

    def crop(self, val, axis):
        if axis == 'x':
            return max(0, min(val, self.width-1))
        elif axis == 'y':
            return max(0, min(val, self.height-1))

    def local_view(self, pos, radius=3):
        ''' Returns a state representation of the position's surroundings '''
        ax, ay = pos
        view = (self.grid[self.crop(ay-radius//2, 'y')]    [self.crop(ax-radius//2, 'x'):self.crop(ax+radius//2+1, 'x')],
                self.grid[self.crop(ay, 'y')]              [self.crop(ax-radius//2, 'x'):self.crop(ax+radius//2+1, 'x')],
                self.grid[self.crop(ay+radius//2, 'y')]    [self.crop(ax-radius//2, 'x'):self.crop(ax+radius//2+1, 'x')]
        )
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
    layout = Layout(19,15)
    layout.generate(200)
    [print(row) for row in layout.grid]
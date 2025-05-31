import heapq
import itertools

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = self.f = self.h = 0
        
    def __eq__(self, other):
        return self.position[0] == other.position[0] and self.position[1] == other.position[1]

def manhattan(pos_a, pos_b):
    return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])

class AStar:
    @staticmethod
    def search(layout, start_pos, goal_pos):
        open_list = []
        closest_set = set()
        g_score = {}
        
        start_node = Node(tuple(layout.to_grid(start_pos)))
        goal_node = Node(tuple(layout.to_grid(goal_pos)))

        counter = itertools.count()
        heapq.heappush(open_list, (start_node.f, next(counter), start_node))
        g_score[start_node.position] = 0
        
        while open_list:
            _, _, current_node = heapq.heappop(open_list)

            if current_node == goal_node:
                path = AStar.reconstruct_path(current_node)
                return [layout.to_real(pos) for pos in path]
            
            closest_set.add(tuple(current_node.position))
            
            for neighbor_pos in AStar.get_neighbors(current_node.position, layout):
                if neighbor_pos in closest_set:
                    continue
                
                tentative_g = current_node.g + 1
                # no need to continue if we know the node with a lower g score
                if neighbor_pos in g_score and tentative_g >= g_score[neighbor_pos]:
                    continue
                
                g_score[neighbor_pos] = tentative_g
                
                neighbor_node = Node(neighbor_pos, parent=current_node)
                neighbor_node.g = current_node.g + 1
                neighbor_node.h = manhattan(neighbor_pos, goal_node.position)
                neighbor_node.f = neighbor_node.g + neighbor_node.h
                
                heapq.heappush(open_list, (neighbor_node.f, next(counter), neighbor_node))
        
        return []
        
    @staticmethod
    def reconstruct_path(node):
        path = []
        while node:
            path.append(node.position)
            node = node.parent
        return path[::-1]

    @staticmethod
    def get_neighbors(pos, layout):
        x, y = pos
        directions = [(1,0), (-1,0), (0,1), (0,-1)]
        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < layout.width and 0 <= ny < layout.height:
                if layout.passable((nx * layout.step, ny * layout.step)):
                    neighbors.append((nx,ny))
        return neighbors
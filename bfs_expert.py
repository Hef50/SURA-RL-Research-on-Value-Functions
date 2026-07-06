from collections import deque
from environment import Action

def bfs(maze, start, goal):
    start = tuple(start)
    goal = tuple(goal)

    queue = deque([start]) # double ended queue, puts tuple in list to prevent unwrapping
    visited = {start} # creates a set -> collection of unique items so we don't go in circles
    parent = {} # dictionary of cell -> prev

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while len(queue) > 0:
        # take new item from queue
        current = queue.popleft()

        # if current is the goal, reconstruct path
        if current == goal:
            path = []
            while current in parent:
                path.append(current)
                current = parent[current]
            path.append(start)
            path.reverse()
            return path
        
        for dr, dc in directions:
            nr, nc = current[0] + dr, current[1] + dc
            neighbor = (nr, nc)
            if 0 <= nr < maze.shape[0] and 0 <= nc < maze.shape[1]:
                # if open cell and not visited, add to queue
                if maze[nr, nc] == 0 and neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)
    return None

def generate_expert_actions(path):
    if path is None or len(path) < 2: # if there is no path or path is just starting square
        return [Action.STOP]
    
    actions = []

    delta_to_action = {
        (-1, 0): Action.UP,
        (1, 0): Action.DOWN,
        (0, -1): Action.LEFT,
        (0, 1): Action.RIGHT
    }

    for i in range(len(path) - 1): # len - 1 bc it loops through each element (i) and the next one (i + 1)
        current_cell = path[i]
        next_cell = path[i + 1]
        delta = (next_cell[0] - current_cell[0], next_cell[1] - current_cell[1])
        actions.append(delta_to_action[delta])
    
    actions.append(Action.STOP)
    return actions
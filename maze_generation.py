import numpy as np
import random 
import matplotlib.pyplot as plt

def find_neighbors(x, y, D):
    neighbors = []
    directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < D and 0 <= ny < D:
            neighbors.append((nx, ny))
    return neighbors

def find_in_maze_neighbors(x, y, D, M):
    neighbors = find_neighbors(x, y, D)
    # in_maze_neighbors = []
    # for nx, ny in neighbors:
    #     if M[nx, ny] == 0:
    #         in_maze_neighbors.append((nx, ny))
    # return in_maze_neighbors
    return [(nx, ny) for nx, ny in neighbors if M[nx, ny] == 0]

def generate_maze(D, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    # starting with grid of all ones (walls)
    maze = np.ones((D, D), dtype=int)

    # start Primms at (1, 1) - it shouldn't matter, maze should be possible for any starting point
    start_x, start_y = 1, 1
    maze[start_x, start_y] = 0 

    frontier = []
    frontier.extend(find_neighbors(start_x, start_y, D))

    while len(frontier) > 0:
        random_idx = np.random.randint(0, len(frontier))
        fx, fy = frontier.pop(random_idx)
        if maze[fx, fy] == 1:
            maze[fx, fy] = 0
            mx, my = random.choice(find_in_maze_neighbors(fx, fy, D, maze))
            wall_x = (fx + mx) // 2
            wall_y = (fy + my) // 2
            maze[wall_x, wall_y] = 0
            frontier_neighbors = find_neighbors(fx, fy, D)
            for nx, ny in frontier_neighbors:
                if maze[nx, ny] ==1 and (nx, ny) not in frontier:
                    frontier.append((nx, ny))

    return maze

def simple_visualize(maze, start_pos=None, goal_pos=None):
    plt.imshow(maze, cmap="binary")

    if start_pos is not None:
        row, col = start_pos
        # flip row and col bc x, y
        plt.scatter(col, row, color="green", marker="+", s=200, label="Start")

    if goal_pos is not None:
        row, col = goal_pos
        plt.scatter(col, row, color="gold", marker="*", s=300, label="Goal")
    
    plt.axis("off")
    if start_pos or goal_pos:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.show()

def place_start_goal(maze):
    open_cells = np.argwhere(maze == 0)
    open_tuples = [tuple(cell) for cell in open_cells]
    selected = random.sample(open_tuples, 2)

    start_pos = selected[0]
    goal_pos = selected[1]

    return start_pos, goal_pos

def generate_mazes(n, D):
    dataset = []
    for i in range(n):
        maze = generate_maze(D)
        start, goal = place_start_goal(maze)

        dataset.append({
            "maze": maze,
            "start_pos": start,
            "goal_pos": goal
        })
    return dataset

if __name__ == "__main__":
    D = 10
    print(f"Generating a {D}x{D} maze: ")

    maze_grid = generate_maze(D, seed=42)
    print("Generated Maze Grid Matrix: \n", maze_grid)
    start, goal = place_start_goal(maze_grid)
    simple_visualize(maze_grid, start, goal)
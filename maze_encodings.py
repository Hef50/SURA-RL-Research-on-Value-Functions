import numpy as np

def encode_as_channels(maze, agent_pos, goal_pos):
    channel_walls = maze.flatten().astype(np.float32)

    agent_grid = np.zeros_like(maze, dtype = np.float32)
    agent_grid[agent_pos[0], agent_pos[1]] = 1.0
    channel_agent = agent_grid.flatten()

    goal_grid = np.zeros_like(maze, dtype = np.float32)
    goal_grid[goal_pos[0], goal_pos[1]] = 1.0
    channel_goal = goal_grid.flatten()

    return np.concatenate([channel_walls, channel_agent, channel_goal])

def encode_as_single_array(maze, agent_pos, goal_pos):
    
    grid = maze.copy().astype(np.float32)
    grid[agent_pos[0], agent_pos[1]] = 2.0
    grid[goal_pos[0], goal_pos[1]] = 3.0
    channel = grid.flatten()
    return channel

def encode_as_2d_channels(maze, agent_pos, goal_pos):
    """Returns a 3-channel 2D spatial numpy array of shape (3, D, D)"""
    channel_walls = maze.astype(np.float32)

    channel_agent = np.zeros_like(maze, dtype=np.float32)
    channel_agent[agent_pos[0], agent_pos[1]] = 1.0

    channel_goal = np.zeros_like(maze, dtype=np.float32)
    channel_goal[goal_pos[0], goal_pos[1]] = 1.0

    # Stack them along the first axis to create a (3, D, D) matrix
    return np.stack([channel_walls, channel_agent, channel_goal], axis=0)
import numpy as np

def encode_as_channels(maze, agent_pos, goal_pos):
    channel_walls = maze.flatten().astype(np.float32)

    agent_grid = np.zeros_like(maze, dtype = np.float32) # array with same dimensions as input
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
    # Returns a 3-channel 2D spatial numpy array of shape (3, D, D)
    channel_walls = maze.astype(np.float32)

    channel_agent = np.zeros_like(maze, dtype=np.float32)
    channel_agent[agent_pos[0], agent_pos[1]] = 1.0

    channel_goal = np.zeros_like(maze, dtype=np.float32)
    channel_goal[goal_pos[0], goal_pos[1]] = 1.0

    # Stack them along the first axis to create a (3, D, D) matrix
    return np.stack([channel_walls, channel_agent, channel_goal], axis=0)

def encode_batch(mazes, agents, goals):
    # vectorized version of encode_as_2d_channels -> encodes a whole batch of rollouts at once
    # mazes: (N, D, D), agents/goals: (N, 2). returns (N, 3, D, D) float32
    # this is what lets us run one big forward pass per timestep instead of N tiny batch-1 ones
    N, D, _ = mazes.shape
    walls = mazes.astype(np.float32) # wall channel is literally just the maze

    # one-hot the agent + goal cells across the whole batch with fancy indexing
    idx = np.arange(N)
    channel_agent = np.zeros((N, D, D), dtype=np.float32)
    channel_agent[idx, agents[:, 0], agents[:, 1]] = 1.0

    channel_goal = np.zeros((N, D, D), dtype=np.float32)
    channel_goal[idx, goals[:, 0], goals[:, 1]] = 1.0

    # stack along axis=1 so channels sit right after the batch dim -> (N, 3, D, D)
    return np.stack([walls, channel_agent, channel_goal], axis=1)
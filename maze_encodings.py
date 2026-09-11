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

def encode_as_2d_channels(maze, agent_pos, goal_pos, t=None, max_steps=None):
    # Returns a (3, D, D) stack, or (4, D, D) when a timestep is supplied
    channel_walls = maze.astype(np.float32)

    channel_agent = np.zeros_like(maze, dtype=np.float32)
    channel_agent[agent_pos[0], agent_pos[1]] = 1.0

    channel_goal = np.zeros_like(maze, dtype=np.float32)
    channel_goal[goal_pos[0], goal_pos[1]] = 1.0

    channels = [channel_walls, channel_agent, channel_goal]
    if t is not None:
        # 4th channel: a CONSTANT D x D plane of t / max_steps. Constant because the policy
        # needs a scalar ("how much time is left") and a conv net can only read a scalar if it
        # is broadcast across the grid it convolves over.
        # Deliberately NOT Manhattan distance - that is inferable from the agent and goal
        # planes, and handing it to the value function defeats the purpose of learning one.
        channels.append(np.full_like(channel_walls, np.float32(t) / np.float32(max_steps)))

    # Stack along the first axis -> (C, D, D)
    return np.stack(channels, axis=0)

def encode_batch(mazes, agents, goals, t=None, max_steps=None):
    # vectorized version of encode_as_2d_channels -> encodes a whole batch of rollouts at once
    # mazes: (N, D, D), agents/goals: (N, 2). returns (N, C, D, D) float32, C = 3 or 4
    # this is what lets us run one big forward pass per timestep instead of N tiny batch-1 ones
    N, D, _ = mazes.shape
    walls = mazes.astype(np.float32) # wall channel is literally just the maze

    # one-hot the agent + goal cells across the whole batch with fancy indexing
    idx = np.arange(N)
    channel_agent = np.zeros((N, D, D), dtype=np.float32)
    channel_agent[idx, agents[:, 0], agents[:, 1]] = 1.0

    channel_goal = np.zeros((N, D, D), dtype=np.float32)
    channel_goal[idx, goals[:, 0], goals[:, 1]] = 1.0

    channels = [walls, channel_agent, channel_goal]
    if t is not None:
        # same constant t/max_steps plane as the single-maze encoder - every rollout in the
        # batch is at the SAME timestep t - VecMazeEnv.steps is one global counter, and
        # finished rollouts are frozen by the done mask, so a single scalar is correct here
        channels.append(np.full((N, D, D), np.float32(t) / np.float32(max_steps)))

    # stack along axis=1 so channels sit right after the batch dim -> (N, C, D, D)
    return np.stack(channels, axis=1)
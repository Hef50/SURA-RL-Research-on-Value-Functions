import numpy as np
from maze_generation import generate_maze, place_start_goal

from enum import IntEnum

# Action inherits from IntEnums
class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STOP = 4

class MazeEnv:
    def __init__(self, D=5, max_steps=50):
        self.D = D
        self.max_steps = max_steps
        self.reset()
    
    def reset(self):
        self.maze = generate_maze(self.D)
        self.start_pos, self.goal_pos = place_start_goal(self.maze)

        self.agent_pos = self.start_pos
        self.steps = 0

        return (self.maze, self.agent_pos, self.goal_pos)

    def step(self, action):
        action = Action(action) # turns number into enum type

        self.steps += 1
        reward = 0
        done = False
        
        moves = {
            Action.UP: (-1, 0),
            Action.DOWN: (1, 0),
            Action.LEFT: (0, -1),
            Action.RIGHT: (0, 1)
        }

        if action == Action.STOP:
            if tuple(self.agent_pos) == tuple(self.goal_pos):
                reward = 1
            else:
                reward = 0
            done = True
        else:
            dr, dc = moves[action]
            new_row = self.agent_pos[0] + dr
            new_col = self.agent_pos[1] + dc

            if 0 <= new_row < self.D and 0 <= new_col < self.D:
                # only move if valid open path -> maybe change to fail immediately later
                if self.maze[new_row, new_col] == 0:
                    self.agent_pos = (new_row, new_col)
        
        if self.steps >= self.max_steps:
            done = True
        
        state = (self.maze, self.agent_pos, self.goal_pos)
        return state, reward, done


class VecMazeEnv:
    # batched maze env -> steps N independent rollouts at once with numpy so the training /
    # eval loop can do a single big forward pass per timestep instead of N tiny batch-1 calls
    # move deltas line up with Action indices: UP, DOWN, LEFT, RIGHT (STOP=4 is handled apart)
    MOVES = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])

    def __init__(self, mazes, starts, goals, max_steps):
        self.mazes = np.stack(mazes).astype(np.int8)   # (N, D, D) obstacle grids
        self.agent = np.array(starts, dtype=np.int64)  # (N, 2) current positions
        self.goal = np.array(goals, dtype=np.int64)    # (N, 2) goal positions
        self.N = self.mazes.shape[0]
        self.D = self.mazes.shape[1]
        self.max_steps = max_steps
        self.steps = 0
        self.done = np.zeros(self.N, dtype=bool) # boolean completion trackers managed per-rollout

    def step(self, actions):
        # actions: (N,) int array. returns raw reward (N,) float32 and done (N,) bool
        # same rules as MazeEnv.step, just vectorized across the batch
        reward = np.zeros(self.N, dtype=np.float32)
        active = ~self.done # only rollouts that haven't finished get to act this step

        # STOP branch -> reward 1 only if we're standing on the goal, else 0; ends the rollout either way
        stop = active & (actions == 4)
        on_goal = np.all(self.agent == self.goal, axis=1)
        reward[stop & on_goal] = 1.0
        self.done[stop] = True

        # movement branch -> figure out target cells, only actually move if in-bounds AND not a wall
        # (same "only move if valid open path" rule as the single-env version)
        move = active & (actions < 4)
        target = self.agent + self.MOVES[np.clip(actions, 0, 3)]
        in_bounds = (
            (target[:, 0] >= 0) & (target[:, 0] < self.D) &
            (target[:, 1] >= 0) & (target[:, 1] < self.D)
        )
        valid = move & in_bounds
        # only look up walls for the valid rows so we never index out of range
        rows = np.where(valid)[0]
        open_cell = np.zeros(self.N, dtype=bool)
        open_cell[rows] = self.mazes[rows, target[rows, 0], target[rows, 1]] == 0
        do_move = valid & open_cell
        self.agent[do_move] = target[do_move]

        # global step cap -> once we hit it everything is done (same as the single-env version)
        self.steps += 1
        if self.steps >= self.max_steps:
            self.done[:] = True

        return reward, self.done.copy()


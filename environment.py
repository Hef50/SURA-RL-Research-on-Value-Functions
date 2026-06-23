import numpy as np
from maze_generation import generate_maze, place_start_goal

from enum import IntEnum

class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STOP = 4

class MazeEnv:
    def __init__(self, D=5):
        self.D = D
        self.max_steps = 20
        self.reset()
    
    def reset(self):
        self.maze = generate_maze(self.D)
        self.start_pos, self.goal_pos = place_start_goal(self.maze)

        self.agent_pos = self.start_pos
        self.steps = 0

        return (self.maze, self.agent_pos, self.goal_pos)

    def step(self, action):
        action = Action(action)

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


import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

from environment import MazeEnv
from maze_encodings import encode_as_2d_channels
from model import MazeCNN

# Setup hardware accelerator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_trajectory(env, path, title_text, save_path):
    # Renders a clear 2D grid layout showing walls, start, goal, and the agent's path
    maze = env.maze
    D = maze.shape[0]
    
    # Initialize an RGB matrix (White for empty floor paths)
    grid = np.ones((D, D, 3))
    
    # Draw walls (Black squares)
    for r in range(D):
        for c in range(D):
            if maze[r, c] == 1:
                grid[r, c] = [0.1, 0.1, 0.1] # almost block
                
    # Mark Start (Blue) and Goal (Green)
    start_r, start_c = env.start_pos if hasattr(env, 'start_pos') else (1, 1)
    goal_r, goal_c = env.goal_pos
    grid[start_r, start_c] = [0.2, 0.4, 0.9] # green
    grid[goal_r, goal_c] = [0.1, 0.8, 0.2] # blue
    
    fig, ax = plt.subplots(figsize=(6, 6))
    # maps 2D array to pixel grid, upper origin -> match standard matrix array indexing than traditional math planes
    ax.imshow(grid, origin='upper')
    
    # overlay the agent's exact path steps using a red directional line
    if len(path) > 1:
        path_r = [pos[0] for pos in path]
        path_c = [pos[1] for pos in path]
        ax.plot(path_c, path_r, color='red', linewidth=3, marker='o', markersize=5, label='Agent Path')
        
    ax.set_title(title_text, fontsize=12, fontweight='bold')
    ax.set_xticks(range(D))
    ax.set_yticks(range(D))
    ax.grid(color='gray', linestyle='--', linewidth=0.5)
    
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved visualization trace to: {save_path}")

def run_visualization_suite():
    D = 8
    MAX_STEPS = 50
    
    # Define the weights files you want to inspect qualitatively
    checkpoints = {
        "Early_Stage": "BFS_BC_CNN-RL-starter.pth", # Behavior Cloning baseline checkpoint
        "Late_Stage": "maze_REINFORCE.pth"                # Dynamically scales to match your active group algorithm name
    }
    
    # Instantiate a clean visualization environment
    env = MazeEnv(D=D, max_steps=MAX_STEPS)
    env.reset()
    
    # Lock down the layout structure so both models face the exact same test maze
    test_maze = np.copy(env.maze)
    start_pos = tuple(env.agent_pos)
    goal_pos = tuple(env.goal_pos)
    
    model = MazeCNN(d=D, hidden_dim=128).to(device)
    
    for stage_name, weight_file in checkpoints.items():
        if not os.path.exists(weight_file):
            print(f"Skipping {stage_name}: Checkpoint file '{weight_file}' not found.")
            continue
            
        model.load_state_dict(torch.load(weight_file, map_location=device), strict=False)
        model.eval()
        
        # Reset environment
        env.maze = np.copy(test_maze)
        env.agent_pos = list(start_pos)
        env.goal_pos = list(goal_pos)
        env.steps = 0
        
        path_history = [tuple(start_pos)]
        done = False
        steps = 0
        
        with torch.no_grad():
            while not done and steps < MAX_STEPS:
                state_matrix = encode_as_2d_channels(env.maze, env.agent_pos, env.goal_pos)
                state_tensor = torch.tensor(state_matrix, dtype=torch.float32).unsqueeze(0).to(device)
                
                logits, _ = model(state_tensor)
                
                # execute actions greedily to inspect the deterministic target policy choices
                action = torch.argmax(F.softmax(logits, dim=-1), dim=-1).item()
                
                _, _, done = env.step(action)
                path_history.append(tuple(env.agent_pos))
                steps += 1
                
        # generate the visual 2D grid image file
        title = f"Policy Trajectory - {stage_name}\nSteps Taken: {steps} | Success: {tuple(env.agent_pos) == tuple(env.goal_pos)}"
        filename = f"trajectory_{stage_name.lower()}.png"
        plot_trajectory(env, path_history, title, filename)

if __name__ == "__main__":
    run_visualization_suite()
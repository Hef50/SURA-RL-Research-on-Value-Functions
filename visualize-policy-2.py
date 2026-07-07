import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

from environment import MazeEnv
from maze_encodings import encode_as_2d_channels
from model import MazeCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InteractiveMazeApplet:
    def __init__(self, d=8, max_steps=50):
        self.D = d
        self.MAX_STEPS = max_steps
        
        self.env = MazeEnv(D=self.D, max_steps=self.MAX_STEPS)
        
        self.early_model = MazeCNN(d=self.D, hidden_dim=128).to(device)
        self.late_model = MazeCNN(d=self.D, hidden_dim=128).to(device)
        
        # load checkpoints
        self.load_weights(self.early_model, "BFS_BC_CNN-RL-starter.pth", "Early Stage (BC Starter)")
        self.load_weights(self.late_model, "maze_REINFORCE.pth", "Late Stage (GRPO/RL)") # Change string to match your checkpoint
        
        # set up the dual matplotlib subplots figure layout
        self.fig, (self.ax_early, self.ax_late) = plt.subplots(1, 2, figsize=(12, 6))
        
        # connect the keyboard press callback event listener to the canvas object
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        print("\n=== INTERACTIVE POLICY APPLET INITIATED ===")
        print("-> Press [SPACEBAR] inside the plot window to generate a new maze and compare policies.")
        print("-> Close the window to exit.\n")
        
        # run initial layout generation cycle right away on launch
        self.generate_and_render_comparison()
        plt.show()

    def load_weights(self, model, weight_path, name):
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
            model.eval()
            print(f"Successfully loaded {name} from {weight_path}")
        else:
            print(f"Warning: {weight_path} not found! {name} will use randomized weights.")
            model.eval()

    def roll_out_policy(self, model, maze_structure, start_pos, goal_pos):
        """Executes a greedy trajectory rollout for a given model network."""
        self.env.maze = np.copy(maze_structure)
        self.env.agent_pos = list(start_pos)
        self.env.goal_pos = list(goal_pos)
        self.env.steps = 0
        
        path = [tuple(start_pos)]
        done = False
        steps = 0
        
        with torch.no_grad():
            while not done and steps < self.MAX_STEPS:
                state_matrix = encode_as_2d_channels(self.env.maze, self.env.agent_pos, self.env.goal_pos)
                state_tensor = torch.tensor(state_matrix, dtype=torch.float32).unsqueeze(0).to(device)
                
                logits, _ = model(state_tensor)
                action = torch.argmax(F.softmax(logits, dim=-1), dim=-1).item()
                
                _, _, done = self.env.step(action)
                path.append(tuple(self.env.agent_pos))
                steps += 1
                
        success = tuple(self.env.agent_pos) == tuple(goal_pos)
        return path, steps, success

    def render_axes(self, ax, maze, path, start_pos, goal_pos, title_text):
        ax.clear()
        
        # construct RGB grid representation matrix
        grid = np.ones((self.D, self.D, 3))
        for r in range(self.D):
            for c in range(self.D):
                if maze[r, c] == 1:
                    grid[r, c] = [0.1, 0.1, 0.1] # Dark grey walls
                    
        # mark Blue Start and Green Goal
        grid[start_r := start_pos[0], start_c := start_pos[1]] = [0.2, 0.4, 0.9]
        grid[goal_r := goal_pos[0], goal_c := goal_pos[1]] = [0.1, 0.8, 0.2]
        
        ax.imshow(grid, origin='upper')
        
        # Plot agent's movement step traces
        if len(path) > 1:
            path_r = [pos[0] for pos in path]
            path_c = [pos[1] for pos in path]
            ax.plot(path_c, path_r, color='red', linewidth=3, marker='o', markersize=4)
            
        ax.set_title(title_text, fontsize=10, fontweight='bold')
        ax.set_xticks(range(self.D))
        ax.set_yticks(range(self.D))
        ax.grid(color='gray', linestyle='--', linewidth=0.5)

    def generate_and_render_comparison(self):
        self.env.reset()
        maze_structure = np.copy(self.env.maze)
        start_pos = tuple(self.env.start_pos) if hasattr(self.env, 'start_pos') else tuple(self.env.agent_pos)
        goal_pos = tuple(self.env.goal_pos)
        
        # simulate trajectories across both model check points
        early_path, early_steps, early_ok = self.roll_out_policy(self.early_model, maze_structure, start_pos, goal_pos)
        late_path, late_steps, late_ok = self.roll_out_policy(self.late_model, maze_structure, start_pos, goal_pos)
        
        # draw Early Stage Subplot Viewport
        early_title = f"Early Stage (BC Starter Baseline)\nSteps: {early_steps} | Reached Goal: {early_ok}"
        self.render_axes(self.ax_early, maze_structure, early_path, start_pos, goal_pos, early_title)
        
        # draw Late Stage Subplot Viewport
        late_title = f"Late Stage (RL/GRPO Fine-Tuning)\nSteps: {late_steps} | Reached Goal: {late_ok}"
        self.render_axes(self.ax_late, maze_structure, late_path, start_pos, goal_pos, late_title)
        
        # force the user-interface window canvas panel to refresh instantly
        self.fig.canvas.draw()

    def on_key_press(self, event):
        # check if the triggered key token matches the spacebar
        if event.key == ' ':
            self.generate_and_render_comparison()

if __name__ == "__main__":
    InteractiveMazeApplet(d=8, max_steps=50)
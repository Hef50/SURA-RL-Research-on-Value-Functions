import torch
import random
import numpy as np
from environment import MazeEnv
from maze_encodings import encode_as_channels
from model import MazeMLP

def manhatten_dist(pos_x, pos_y):
    return abs(pos_x[0] - pos_y[0]) + abs(pos_x[1] - pos_y[1])

def evaluate_random_policy(num_mazes=100, D=5):
    env = MazeEnv(D=D)
    successes = 0
    total_final_distance = 0

    for _ in range(num_mazes):
        env.reset()
        done = False
        steps = 0
        
        while not done and steps < 20:
            random_action = random.randint(0, 4)
            _, _, done = env.step(random_action)
            steps += 1
        
        if done and env.agent_pos == env.goal_pos:
            successes += 1
        
        total_final_distance += manhatten_dist(env.agent_pos, env.goal_pos)
    
    success_rate = (successes / num_mazes) * 100
    avg_distance = total_final_distance / num_mazes

    print(f"Random Policy Baseline:")
    print(f"  Success Rate: {success_rate:.1f}%")
    print(f"  Avg Final Distance to Goal: {avg_distance:.2f} cells\n")

def evaluate_model_policy(model_path, num_mazes=100, D=5):
    env = MazeEnv(D=D)

    env.reset()
    sample_state = encode_as_channels(env.maze, env.agent_pos, env.goal_pos)
    input_dim = len(sample_state)

    model = MazeMLP(input_dim=input_dim, hidden_dim=128)
    # torch.load(model_path) -> load saved .pth file from computer mem as state ict
    model.load_state_dict(torch.load(model_path))

    model.eval()

    successes = 0
    total_final_distance = 0

    with torch.no_grad():
        for _ in range(num_mazes):
            env.reset()
            done = False
            steps = 0

            while not done and steps < 20:
                state_vector = encode_as_channels(env.maze, env.agent_pos, env.goal_pos)
                state_tensor = torch.tensor(np.array([state_vector]), dtype=torch.float32)
                logits = model(state_tensor)
                action = torch.argmax(logits, dim=1).item()
                _, _, done = env.step(action)
                steps += 1
        
            if done and env.agent_pos == env.goal_pos:
                successes += 1
            
            total_final_distance += manhatten_dist(env.agent_pos, env.goal_pos)
    
    success_rate = (successes / num_mazes) * 100
    avg_distance = total_final_distance / num_mazes

    print(f"Trained MLP Policy Performance:")
    print(f"  Success Rate: {success_rate:.1f}%")
    print(f"  Avg Final Distance to Goal: {avg_distance:.2f} cells\n")



if __name__ == "__main__":
    D = 5
    print("--- Starting Evaluation Arena ---")

    evaluate_random_policy(num_mazes=100, D=5)
    evaluate_model_policy(model_path="maze_mlp.pth", num_mazes=100, D=5)
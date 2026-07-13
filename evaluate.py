import torch
import torch.nn.functional as F
import random
import numpy as np
from environment import MazeEnv
from maze_encodings import encode_as_channels
from model import MazeMLP

def manhatten_dist(pos_x, pos_y):
    return abs(pos_x[0] - pos_y[0]) + abs(pos_x[1] - pos_y[1])

def evaluate_random_policy(num_mazes=100, D=5, max_steps=50):
    env = MazeEnv(D=D, max_steps=max_steps)
    successes = 0
    total_final_distance = 0

    for _ in range(num_mazes):
        env.reset()
        done = False
        steps = 0
        
        while not done and steps < max_steps:
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
    return success_rate

def evaluate_model_policy_greedy(model, env, encoding_fn, num_mazes=100, max_steps=50, modeltype="CNN"):
    # Evaluates model based on deterministic greedy argmax choices
    model.eval()

    # Each pytorch tensor has a .device() method, so we're going to 
    # the first tensor of the model's parameters (NEXT to the model header) to find its device
    device = next(model.parameters()).device # takes 

    successes = 0
    total_final_distance = 0

    with torch.no_grad():
        for _ in range(num_mazes):
            env.reset()
            done = False
            steps = 0

            while not done and steps < max_steps:  
                if modeltype == "MLP":
                    state_vector = encode_as_channels(env.maze, env.agent_pos, env.goal_pos)
                    state_tensor = torch.tensor(np.array([state_vector]), dtype=torch.float32).to(device)
                elif modeltype == "CNN":
                    state_matrix = encoding_fn(env.maze, env.agent_pos, env.goal_pos)
                    state_tensor = torch.tensor(state_matrix, dtype=torch.float32).unsqueeze(0).to(device)
                else:
                    raise ValueError(f"Unknown MODEL_TYPE specified: {modeltype}")

                logits, _ = model(state_tensor)
                action = torch.argmax(logits, dim=1).item()
                _, _, done = env.step(action)
                steps += 1
        
            if done and env.agent_pos == env.goal_pos:
                successes += 1
            
            # total_final_distance += manhatten_dist(env.agent_pos, env.goal_pos)
    
    success_rate = (successes / num_mazes) * 100
    # avg_distance = total_final_distance / num_mazes

    # print(f"Trained MLP Policy Performance:")
    # print(f"  Success Rate: {success_rate:.1f}%")
    # print(f"  Avg Final Distance to Goal: {avg_distance:.2f} cells\n")

    return success_rate

# Measures Pass@k: Pass@k rate: The percentage of mazes where AT LEAST ONE
# out of k stochastic rollouts successfully reaches the goal.
def evaluate_stochastic_pass_k(model, env, encoding_fn, num_mazes=100, N=10, max_steps=50, modeltype="CNN"):
    # Evaluates model by random sampling from its policy probability distribution
    model.eval()
    device = next(model.parameters()).device

    successful_mazes = 0

    with torch.no_grad():
        for _ in range(num_mazes):
            env.reset()
            # If env.maze is ever variable, this freezes maze so successive env.reset()s don't create new mazes
            maze_structure = np.copy(env.maze)
            start_pos = tuple(env.agent_pos)
            goal_pos = tuple(env.goal_pos)

            maze_solved = False
            
            for _ in range(N):
                # If env.maze is ever variable, this brings back starting pos
                env.maze = np.copy(maze_structure)
                env.agent_pos = list(start_pos)
                env.goal_pos = list(goal_pos)
                done = False
                steps = 0
                
                while not done and steps < max_steps:
                    # [] for (3D^2) -> (1, 3D^2) for batch processing with unsqueeze
                    # np.array for C-style array contiguous memory allocation
                    # dtype float32 for casting bc that's what model uses
                    # Tensor object cast to track gradients and do fast matrix multiplication and input to model 

                    if modeltype == "MLP":
                        state_vector = encode_as_channels(env.maze, env.agent_pos, env.goal_pos)
                        state_tensor = torch.tensor(np.array([state_vector]), dtype=torch.float32).to(device)
                    elif modeltype == "CNN":
                        state_matrix = encoding_fn(env.maze, env.agent_pos, env.goal_pos)
                        state_tensor = torch.tensor(state_matrix, dtype=torch.float32).unsqueeze(0).to(device)
                    else:
                        raise ValueError(f"Unknown MODEL_TYPE specified: {modeltype}")
                    
                    logits, _ = model(state_tensor)
                    
                    # Convert logits to discrete probability distributions, dim=1 to select actions, not batches (dim=0)
                    probs = F.softmax(logits, dim=1)
                    
                    # Randomly sample an index according to probability array distribution
                    action = torch.multinomial(probs, num_samples=1).item()
                    
                    _, _, done = env.step(action)
                    steps += 1
                    
                if done and env.agent_pos == env.goal_pos:
                    maze_solved = True
                    break
            
            if maze_solved:
                successful_mazes += 1
            
    return (successful_mazes / num_mazes) * 100

# Measures Mean@k: The average execution success rate across all sampled rollouts.
def evaluate_stochastic_mean_k(model, env, encoding_fn, num_mazes=100, N=10, max_steps=50, modeltype="CNN"):
    # Evaluates model by random sampling from its policy probability distribution
    model.eval()
    device = next(model.parameters()).device

    # success rates is an array bc its the average pass rate @ k attempts
    maze_success_rates = [] 

    with torch.no_grad():
        for _ in range(num_mazes):
            env.reset()
            # If env.maze is ever variable, this freezes maze so successive env.reset()s don't create new mazes
            maze_structure = np.copy(env.maze)
            start_pos = env.agent_pos
            goal_pos = env.goal_pos
            
            successful_attempts = 0
            
            for _ in range(N):
                # If env.maze is ever variable, this brings back starting pos
                env.maze = np.copy(maze_structure)
                env.agent_pos = start_pos
                env.goal_pos = goal_pos
                done = False
                steps = 0
                
                while not done and steps < max_steps:
                    # [] for (3D^2) -> (1, 3D^2) for batch processing with unsqueeze
                    # np.array for C-style array contiguous memory allocation
                    # dtype float32 for casting bc that's what model uses
                    # Tensor object cast to track gradients and do fast matrix multiplication and input to model 

                    if modeltype == "MLP":
                        state_vector = encode_as_channels(env.maze, env.agent_pos, env.goal_pos)
                        state_tensor = torch.tensor(np.array([state_vector]), dtype=torch.float32).to(device)
                    elif modeltype == "CNN":
                        state_matrix = encoding_fn(env.maze, env.agent_pos, env.goal_pos)
                        state_tensor = torch.tensor(state_matrix, dtype=torch.float32).unsqueeze(0).to(device)
                    else:
                        raise ValueError(f"Unknown MODEL_TYPE specified: {modeltype}")
                    
                    logits, _ = model(state_tensor)
                    
                    # Convert logits to discrete probability distributions, dim=1 to select actions, not batches (dim=0)
                    probs = F.softmax(logits, dim=1)
                    
                    # Randomly sample an index according to probability array distribution
                    action = torch.multinomial(probs, num_samples=1).item()
                    
                    _, _, done = env.step(action)
                    steps += 1
                    
                if done and env.agent_pos == env.goal_pos:
                    successful_attempts += 1
            
            # Record the execution fraction that reached the target for this maze
            maze_success_rates.append(successful_attempts / N)
            
    return np.mean(maze_success_rates) * 100


if __name__ == "__main__":
    # Standard 8x8 sandbox arena execution code block for local validation testing
    D = 8
    MAX_STEPS = 50
    HIDDEN_DIM = 128
    print(f"--- Launching 8x8 Evaluation Arena Context Loop ---")
    
    rand_rate = evaluate_random_policy(num_mazes=100, D=D, max_steps=MAX_STEPS)
    print(f"Random Policy Success: {rand_rate:.1f}%")
    
    # Setup baseline model verification structures
    env = MazeEnv(D=D, max_steps=MAX_STEPS)
    env.reset()
    sample_state = encode_as_channels(env.maze, env.agent_pos, env.goal_pos)
    
    model = MazeMLP(input_dim=len(sample_state), hidden_dim=HIDDEN_DIM, num_layers=3)
    try:
        model.load_state_dict(torch.load("maze_mlp.pth"))
        print("Loaded weights from maze_mlp.pth successfully.")
        
        greedy_rate = evaluate_model_policy_greedy(model, env, encode_as_channels, num_mazes=100, max_steps=MAX_STEPS)
        stoch_rate = evaluate_stochastic_pass_k(model, env, encode_as_channels, num_mazes=100, N=10, max_steps=MAX_STEPS)
        
        print(f"Greedy Policy Success Rate: {greedy_rate:.1f}%")
        print(f"Stochastic Pass@10 Success Rate: {stoch_rate:.1f}%")
    except FileNotFoundError:
        print("Could not find weights file. Run train.py first to create maze_mlp.pth.")
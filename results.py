import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import tkinter as tk
from tkinter import ttk

from environment import MazeEnv
from model import MazeMLP
from train import collect_expert_data, MazeDataset
from evaluate import manhatten_dist, evaluate_random_policy
from maze_encodings import encode_as_channels, encode_as_single_array

def run_experiment(encoding_name, encoding_fn, hidden_dim, num_train_mazes=400, num_val_mazes=100, epochs=15, batch_size=32, D=5):
    train_states, train_actions = collect_expert_data(num_mazes=num_train_mazes, D=D, encoding_fn=encoding_fn)
    val_states, val_actions = collect_expert_data(num_mazes=num_val_mazes, D=D, encoding_fn=encoding_fn)

    train_loader = DataLoader(MazeDataset(train_states, train_actions), batch_size=batch_size, shuffle=True)
    input_dim = train_states.shape[1]
    model = MazeMLP(input_dim=input_dim, hidden_dim=hidden_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for batch_states, batch_actions in train_loader:
            optimizer.zero_grad()
            logits = model(batch_states)
            loss = criterion(logits, batch_actions)
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        train_logits = model(train_states)
        train_preds = torch.argmax(train_logits, dim=1)
        train_acc = (train_preds == train_actions).sum().item() / len(train_states) * 100
    
    env = MazeEnv(D=D)
    successes = 0

    with torch.no_grad():
        for _ in range(num_val_mazes):
            env.reset()
            done = False
            steps = 0

            while not done and steps < 20:
                state_vector = encoding_fn(env.maze, env.agent_pos, env.goal_pos)
                state_tensor = torch.tensor(np.array([state_vector]), dtype=torch.float32)
                logits = model(state_tensor)
                action = torch.argmax(logits, dim=1).item()
                _, _, done = env.step(action)
                steps += 1

            if done and env.agent_pos == env.goal_pos:
                successes += 1

    val_success_rate = (successes / num_val_mazes) * 100
    return train_acc, val_success_rate

def get_random_baseline_success(num_mazes=100, D=5):
    env = MazeEnv(D=D)
    successes = 0
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
    return (successes / num_mazes) * 100

def display_gui_results_table(random_floor, results_table):
    root = tk.Tk()
    root.title("Results")
    
    columns = ("Policy", "Encoding", "Hidden Size", "Train Accuracy", "Validation Success %")
    tree = ttk.Treeview(root, columns=columns, show="headings", height=6)
    tree.pack(expand=True, fill="both")

    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=130, anchor="center")

    tree.insert("", "end", values=("Random", "———", "———", "———", f"{random_floor:.1f}%"))
    for row in results_table:
        tree.insert("", "end", values=(row["Policy"], row["Encoding"], row["Hidden Size"], row["Train Accuracy"], row["Validation Success"]))

    root.mainloop()

if __name__ == "__main__":
    D = 5 # maze size
    print(" --- Model Summary Evaluation Comparison --- ")

    experiments = [
        {"name": "MLP A (one-hot)", "fn": encode_as_channels,     "size": 64},
        {"name": "MLP A (one-hot)", "fn": encode_as_channels,     "size": 256},
        {"name": "MLP B (integer)", "fn": encode_as_single_array, "size": 64},
        {"name": "MLP B (integer)", "fn": encode_as_single_array, "size": 256},
    ]

    print("Evaluating Random Policy Baseline floor...")
    random_floor = get_random_baseline_success(num_mazes=100, D=5)

    results_table = []

    for exp in experiments:
        print(f"Running Experiment: {exp['name']} | Hidden Size: {exp['size']}...")
        train_acc, val_success = run_experiment(exp['name'], exp['fn'], exp['size'], D=D)

        results_table.append({
            "Policy": exp['name'],
            "Encoding": "one-hot channels" if exp['fn'] == encode_as_channels else "integer grid",
            "Hidden Size": exp['size'],
            "Train Accuracy": f"{train_acc:.1f}%",
            "Validation Success": f"{val_success:.1f}%"
        })

    display_gui_results_table(random_floor, results_table)
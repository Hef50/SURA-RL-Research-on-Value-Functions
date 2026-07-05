import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import wandb

from environment import MazeEnv
from bfs_expert import bfs, generate_expert_actions
from maze_encodings import encode_as_channels, encode_as_single_array, encode_as_2d_channels
from model import MazeMLP, MazeCNN
from evaluate import evaluate_model_policy_greedy, evaluate_stochastic_pass_k

class MazeDataset(Dataset):
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

def collect_expert_data(num_mazes, D, encoding_fn):
    env = MazeEnv(D=D)
    states = []
    actions = []

    for _ in range(num_mazes):
        maze, start, goal = env.reset()
        cell_path = bfs(maze, start, goal)
        
        if cell_path is None:
            continue

        expert_actions = generate_expert_actions(cell_path)

        for action in expert_actions:
            current_state_vector = encoding_fn(env.maze, env.agent_pos, env.goal_pos)
            # add snapshot to states and actions history list
            states.append(current_state_vector)
            actions.append(int(action))
            env.step(action)
    
    # returns (states tensor, actions tensor) of experts
    return torch.tensor(np.array(states), dtype=torch.float32), torch.tensor(np.array(actions), dtype=torch.long)

def train_behavioral_cloning():
    D = 8
    EPOCHS = 20 # solid balance for small grids
    BATCH_SIZE = 32 # efficient batch without overloading mem
    HIDDEN_DIM = 128 # should be enough to learn parmaeters
    LEARNING_RATE = 0.001
    NUM_TRAIN_MAZES = 10000
    NUM_VAL_MAZES = 100 
    MAX_ROLLOUT_STEPS = 50 
    NUM_LAYERS = 2

    wandb.init(
        project="SURA",
        name=f"BC_CNN_{D}x{D}_hd{HIDDEN_DIM}_data10k",
        config={                      
            "grid_size": D,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS, 
            "lr": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "num_train_mazes": NUM_TRAIN_MAZES,
            "encoding": "2d-channels",
            "epochs": EPOCHS,
        },
    )

    print("Collecting expert BFS solves...")

    # ----- MLP -----
    # train = expert states tensor
    # train_states, train_actions = collect_expert_data(num_mazes=400, D=D, encoding_fn=encode_as_channels)
    # for 80-20 cross-validation
    # val_states, val_actions = collect_expert_data(num_mazes=100, D=D, encoding_fn=encode_as_channels)

    # ----- CNN -----
    train_states, train_actions = collect_expert_data(num_mazes=2000, D=D, encoding_fn=encode_as_2d_channels)
    val_states, val_actions = collect_expert_data(num_mazes=100, D=D, encoding_fn=encode_as_2d_channels)

    train_loader = DataLoader(MazeDataset(train_states, train_actions), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(MazeDataset(val_states, val_actions), batch_size=BATCH_SIZE, shuffle=False)

    val_env = MazeEnv(D=D)

    input_dim = train_states.shape[1]
    # model = MazeMLP(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
    model = MazeCNN(d=D, hidden_dim=HIDDEN_DIM)

    criterion = nn.CrossEntropyLoss()
    # Adaptive Moment Estimation, Learning Rate
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    global_step = 0
    LOG_INTERVAL_STEPS = 10  # Track and log metrics every 10 gradient steps

    print(f"Dataset compiled: {len(train_states)} train steps, {len(val_states)} validation steps.")
    print("Beginning training loop optimization...")

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        correct_train = 0
        samples_since_last_log = 0

        for batch_states, batch_actions in train_loader:
            optimizer.zero_grad()

            # training batch has expert data
            logits = model(batch_states)
            # loss var stores pointer to logits and model
            loss = criterion(logits, batch_actions) 

            # loss backward affects gradient accumulators through
            # pointer to logits, which store spointer to hidden layer tensors like model.fc1.weight
            loss.backward()
            optimizer.step()
            global_step += 1 # incremented bc optimizer step is incremented above

            # .item() extracts float loss value 
            # total loss = average loss per sample * number of samaples in batch
            total_train_loss += loss.item() * batch_states.size(0)

            # finds top choice, dim =1 -> look horizontally, not vertically (dim=0)
            preds = torch.argmax(logits, dim=1)
            # sums correctness as int
            correct_train += (preds == batch_actions).sum().item()

            # adds the batch size (or less than the batch size if it's at the end)
            samples_since_last_log += batch_states.size(0)

            if global_step % LOG_INTERVAL_STEPS == 0:
                current_step_loss = loss.item()
                current_step_acc = (preds == batch_actions).float().mean().item() * 100

                wandb.log({
                    "train_loss_step": current_step_loss,
                    "train_acc_step": current_step_acc,
                    "total_samples_seen": global_step * BATCH_SIZE,
                    "global_step": global_step
                })
        
        train_loss = total_train_loss / len(train_states)
        train_acc = (correct_train / len(train_states)) * 100

        # VALIDATION PHASE
        model.eval()
        total_val_loss = 0
        correct_val = 0

        with torch.no_grad():
            for batch_states, batch_actions in val_loader:
                logits = model(batch_states)
                loss = criterion(logits, batch_actions)

                total_val_loss += loss.item() * batch_states.size(0)
                preds = torch.argmax(logits, dim=1)
                correct_val += (preds == batch_actions).sum().item()
            
        val_loss = total_val_loss / len(val_states)
        val_acc = (correct_val / len(val_states)) * 100

        print("Executing physical rollout simulations on validation mazes...")
        greedy_success = evaluate_model_policy_greedy(
            model=model, 
            env=val_env, 
            encoding_fn=encode_as_2d_channels, 
            num_mazes=50, 
            max_steps=MAX_ROLLOUT_STEPS
        )

        stochastic_success = evaluate_stochastic_pass_k(
            model=model, 
            env=val_env, 
            encoding_fn=encode_as_2d_channels, 
            num_mazes=50, 
            N=10, 
            max_steps=MAX_ROLLOUT_STEPS
        )

        # epoch + 1 to number at 1
        # :02d to keep columns aligned to 2 digits
        # .nf = fixed point decimal for n places
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}%")
        print(f"        Rollout Success Rates -> Greedy: {greedy_success:.1f}% | Stochastic Pass@10: {stochastic_success:.1f}%")

        wandb.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "greedy_success_rate": greedy_success,
            "stochastic_success_rate": stochastic_success,
            "epoch": epoch + 1,
            "global_step": global_step
        })
    
    torch.save(model.state_dict(), "maze_mlp.pth")
    print("Model weights successfully saved to maze_mlp.pth!")

    wandb.finish()

if __name__ == "__main__":
    train_behavioral_cloning()
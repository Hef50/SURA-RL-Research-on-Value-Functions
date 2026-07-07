import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import wandb
import collections

from environment import MazeEnv
from maze_encodings import encode_as_2d_channels
from model import MazeCNN
from evaluate import evaluate_model_policy_greedy, evaluate_stochastic_pass_k, evaluate_stochastic_mean_k

# use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using hardware accelerator: {device}")

def train_reinforce():
    # -- HYPERPARAMETERS --
    D = 8 # size of maze
    GAMMA = 0.99 # discount factor for future rewards
    LEARNING_RATE = 0.0005 # Lower LR for RL for policy gradient stability
    TOTAL_EPISODES = 3000 # Total num of envr rollout episodes to train
    MAX_STEPS = 50
    LOG_INTERVAL = 10 # Log interval to W&B -> every 10 episodes
    EVAL_INTERVAL = 100 # Evaluate on held-out test mazes every 100 episodes
    USE_BASELINE = True # Enable baseline critic value function
    CRITIC_COEFF = 0.5 # downscaling critic's dominance to protect policy learning if needed

    wandb.init(
        project="SURA",
        name=f"RL_{f'REINFORCE_Baseline_CC:{CRITIC_COEFF}' if USE_BASELINE else 'Vanilla_REINFORCE'}_{D}x{D}",
        config={
            "grid_size": D,
            "algorithm": "REINFORCE with Baseline" if USE_BASELINE else "Vanilla REINFORCE",
            "gamma": GAMMA,
            "lr": LEARNING_RATE,
            "max_steps": MAX_STEPS,
            "use_baseline": USE_BASELINE,
            "critic_coeff": CRITIC_COEFF if USE_BASELINE else 0.0 
        }
    )

    env = MazeEnv(D=D, max_steps=MAX_STEPS)
    val_env = MazeEnv(D=D, max_steps=MAX_STEPS)

    # Load model to graphics card memory (VRAM)
    model = MazeCNN(d=D, hidden_dim=128).to(device)

    # Loads model to VRAM
    # Strict=false acknowledges that starter doesn't have values for fc_critic, etc. but that's okay
    model.load_state_dict(torch.load("BFS_BC_CNN-RL-starter.pth", map_location=device), strict=False) 
    print("Loaded warm-start maze_CNN baseline.")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    global_step = 0

    # Initialize success tracking window
    success_history = [] 

    print("Beginning on-policy REINFORCE optimization loop...")

    for episode in range(TOTAL_EPISODES):
        model.train() # set to training mode so parameters track autograd updates
        env.reset()
        done = False # boolean completion tracker managed by maze environment
        states, actions, log_probs, rewards, entropies, state_values = [], [], [], [], [], []

        steps = 0 # track local time steps to give MAX_STEPS cutoff

        while not done and steps < MAX_STEPS:
            # (3, 8, 8) snapshot of current state
            state_matrix = encode_as_2d_channels(env.maze, env.agent_pos, env.goal_pos)

            # Cast numpy matrix to float tensor and add a dummy batch dim with unsqueeze -> (1, 3, 8, 8)
            # Allocate input tensor array to GPU before running the forward pass
            state_tensor = torch.tensor(state_matrix, dtype=torch.float32).unsqueeze(0).to(device)

            logits, state_value = model(state_tensor)

            # softmax with dim -1 to choose the last index since logits is (1, 5) 
            probs = F.softmax(logits, dim=-1)

            # create sampling distribution from our probabilities + sample from it
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            # calculate logprob and policy entropy
            # -> this is why we're using torch distribution btw, so we don't have to write logprob and entropy functions ourselves
            # this logprob = pi_theta(a_t | s_t)
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            _, reward, done = env.step(action.item())

            # Append values to trajectory
            log_probs.append(log_prob)
            rewards.append(reward)
            entropies.append(entropy)
            state_values.append(state_value)

            steps += 1
        
        # Calculate discounted reward by iterating in reverse
        discounted_returns = []
        G = 0

        for r in reversed(rewards):
            # Horner's method-style G_t = r_t + gamma * G_{t+1}
            G = r + GAMMA * G
            # insert at 0 to reverse the reversed list to get discounted returns back in regular order (forward-temporal)
            discounted_returns.insert(0, G)
        
        discounted_returns = torch.tensor(discounted_returns, dtype=torch.float32).to(device)
        # Concatenate the list of single-item value tensors of dimension (1,1)
        # into a continuous 1D tensor vector to align dimensionally with the discounted returns 
        state_values = torch.cat(state_values).squeeze(-1) 
        

        policy_loss = []
        value_loss = []

        # Zip collects together logprob and its corresponding return into tuples
        for log_prob, G_t, V_s in zip(log_probs, discounted_returns, state_values):
            if USE_BASELINE:
                advantage = G_t - V_s.item()
                value_loss.append(F.mse_loss(V_s, torch.tensor(G_t, device=device)))
            else:
                advantage = G_t
            
            # since loss = -pi_theta(a_t | s_t) * G_t from policy gradient theorem
            policy_loss.append(-log_prob * advantage)
        
        # CONVENIENCE: torch turns python list into 1D tensor of size (15) so we can call convenient .sum() function
        # IMPORTANCE: creates a master 'StackBackward' intersection node in the computational graph, allowing the trace backward:
        # loss -> sumbackward from sum -> stackbackward from stack -> mulbackward from log_prob * G_t -> categorical from dist
        # -> dist -> probs -> softmax -> logits -> model -> weights
        loss = torch.stack(policy_loss).sum()
        if USE_BASELINE:
            # Sharing loss backprop for efficiency, learning the same representation of the maze
            loss += CRITIC_COEFF * torch.stack(value_loss).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1

        episode_reward = sum(rewards)
        mean_entropy = torch.stack(entropies).mean().item()

        # Success requirements
        is_success = 1.0 if (done and tuple(env.agent_pos) == tuple(env.goal_pos) and episode_reward > 0) else 0.0
        success_history.append(is_success)

        # Create rolling window
        rolling_window = success_history[-100:] # go from last 100 to end
        rolling_success_rate = np.mean(rolling_window) * 100

        if global_step % EVAL_INTERVAL == 0:
            print(f"\n--- Running Three-Metric Validation Suite Checkpoint at Step {global_step} ---")
            val_greedy_rate = evaluate_model_policy_greedy(
                model=model,
                env=val_env,
                encoding_fn=encode_as_2d_channels,
                num_mazes=50,
                max_steps=MAX_STEPS,
                modeltype="CNN"
            )
            
            # Track group optimization footprint for upcoming MaxRL comparisons
            val_pass_k = evaluate_stochastic_pass_k(
                model=model,
                env=val_env,
                encoding_fn=encode_as_2d_channels,
                num_mazes=50,
                N=10,
                max_steps=MAX_STEPS,
                modeltype="CNN"
            )

            val_mean_1 = evaluate_stochastic_mean_k(
                model=model,
                env=val_env,
                encoding_fn=encode_as_2d_channels,
                num_mazes=50,
                N=1,
                max_steps=MAX_STEPS,
                modeltype="CNN"
            )

            print(f"Validation Rates -> Greedy: {val_greedy_rate:.1f}% | Stochastic Pass@10: {val_pass_k:.1f}% | Stochastic Mean@1: {val_mean_1:.1f}%\n")

            wandb.log({
                "val_greedy_success_rate": val_greedy_rate,
                "val_stochastic_pass_10_rate": val_pass_k,
                "val_stochastic_mean_1_rate": val_mean_1,
                "global_step": global_step
            })

        if global_step % LOG_INTERVAL == 0:
            print(f"Episode {global_step:04d} | Reward: {episode_reward:.2f} | Rolling Success: {rolling_success_rate:.1f}% | Steps: {steps} | Entropy: {mean_entropy:.4f}")

            # Critic Loss
            c_loss = torch.stack(value_loss).sum().item() if USE_BASELINE else 0.0

            wandb.log({
                "mean_reward": episode_reward,
                "rolling_train_success_rate": rolling_success_rate,
                "response_length": steps,
                "policy_entropy": mean_entropy,
                "critic_value_loss": c_loss,
                "global_step": global_step
            })
    
    torch.save(model.state_dict(),"maze_REINFORCE.pth")
    print("Vanilla REINFORCE training complete. Weights saved successfully!")

    wandb.finish()

if __name__ == "__main__":
    train_reinforce()        


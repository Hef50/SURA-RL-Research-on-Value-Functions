import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import wandb

from environment import MazeEnv
from maze_encodings import encode_as_2d_channels
from model import MazeCNN
from evaluate import evaluate, EvalMode
from maze_generation import build_fixed_eval_set

# use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using hardware accelerator: {device}")

def train_reinforce():
    # -- HYPERPARAMETERS --
    D = 8 # size of maze
    GAMMA = 0.99 # discount factor for future rewards
    LEARNING_RATE = 5e-5 # Lower LR for RL for policy gradient stability
    TOTAL_UPDATES = 200 # How many gradient steps (how many times optimizer.step() called)
    BATCH_SIZE = 32 # Mazes (independent environments) per update
    GROUP_SIZE = 8 # Rollouts per maze (1 for REINFORCE, __ for group methods)

    MAX_STEPS = 60
    LOG_INTERVAL = 10 # Log interval to W&B -> every __ updates
    EVAL_INTERVAL = 20 # Evaluate on held-out test mazes every __ updates
    CRITIC_COEFF = 0.1 # downscaling critic's dominance to protect policy learning if needed
    ENTROPY_COEFF = 0.01 # Exploration coefficient (beta) to scale the policy entropy bonus, preventing premature mode collapse
    

    ALGORITHM = "REINFORCE"
    USE_BASELINE = True # Enable baseline critic value function

    USE_FIXED_VAL = True
    VAL_SEED = 12345
    NUM_VAL_MAZES = 50
    val_tag = f"valSeed{VAL_SEED}" if USE_FIXED_VAL else "valRandom"
    run_name = (
        f"RL_{ALGORITHM}_G{GROUP_SIZE}_{D}x{D} -{val_tag}"
        if ALGORITHM != "REINFORCE"
        else f"RL_{'REINFORCE_Baseline_CC:' + str(CRITIC_COEFF) if USE_BASELINE else 'Vanilla_REINFORCE'}_{D}x{D}"
    )
    algo_config = ALGORITHM if ALGORITHM != "REINFORCE" else ("REINFORCE_Baseline" if USE_BASELINE else "Vanilla_REINFORCE")

    # Dependent Reference Counts
    TOTAL_ENVS = TOTAL_UPDATES * BATCH_SIZE # Num mazes 
    CURRENT_GROUP = GROUP_SIZE if ALGORITHM != "REINFORCE" else 1
    TOTAL_ROLLOUTS = TOTAL_ENVS * CURRENT_GROUP

    wandb.init(
        project="SURA",
        name=run_name,
        config={
            "grid_size": D,
            "algorithm": algo_config,
            "group_size": GROUP_SIZE if ALGORITHM != "REINFORCE" else 1,
            "gamma": GAMMA,
            "lr": LEARNING_RATE,
            "max_steps": MAX_STEPS,
            "use_baseline": USE_BASELINE,
            "critic_coeff": CRITIC_COEFF if USE_BASELINE else 0.0,
            "entropy_coeff": ENTROPY_COEFF,
            "use_fixed_val": USE_FIXED_VAL,
            "val_seed": VAL_SEED if USE_FIXED_VAL else None,
            "num_val_mazes": NUM_VAL_MAZES,
            "total_envs": TOTAL_ENVS, 
            "total_rollouts": TOTAL_ROLLOUTS
        }
    )

    env = MazeEnv(D=D, max_steps=MAX_STEPS)
    val_env = MazeEnv(D=D, max_steps=MAX_STEPS)
    fixed_val_mazes = (
        build_fixed_eval_set(D=D, num_mazes=NUM_VAL_MAZES, seed=VAL_SEED)
        if USE_FIXED_VAL else None
    )



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

    print("Beginning on-policy RL optimization loop...")

    for update in range(TOTAL_UPDATES):
        model.train() # set to training mode so parameters track autograd updates
        optimizer.zero_grad()

        # batch accumulators
        batch_success, batch_reward, batch_lengths, batch_entropy = [], [], [], []
        batch_c_loss = 0.0

        for b in range(BATCH_SIZE):
            env.reset()
            maze_structure = np.copy(env.maze) # Caches static maze for group rollouts
            start_pos = tuple(env.agent_pos) # stores fixed start coords of agent
            goal_pos = tuple(env.goal_pos)

            group_log_probs = [] # stores lists of step log-probabilities for each group rollout
            group_rewards = [] # stores the accumulated terminal total reward for each group rollout
            group_entropies = [] # stores lists of action selection entropies across the group
            group_state_values = [] # Tracked for REINFORCE with baseline for backwards compatibility
            group_success = [] # for maxRL tracking
            group_step_rewards = [] # for per-step negative reward
            group_lengths = []
            CURRENT_GROUP_SIZE = GROUP_SIZE if ALGORITHM != "REINFORCE" else 1

            for g in range(CURRENT_GROUP_SIZE):
                env.maze = np.copy(maze_structure) # Restores frozen maze obstacle
                env.agent_pos = list(start_pos) # Teleports agent back to start pos
                env.goal_pos = list(goal_pos)
                env.steps = 0 # Resets this for max setp count
                done = False # boolean completion tracker managed by maze environment
            
                log_probs, rewards, entropies, state_values = [], [], [], []
                rollout_success = False # Tracks success of current rollout
                steps = 0 # track local time steps to give MAX_STEPS cutoff

                while not done and steps < MAX_STEPS:
                    # (3, 8, 8) snapshot of current state
                    state_matrix = encode_as_2d_channels(env.maze, env.agent_pos, env.goal_pos)

                    # Cast numpy matrix to float tensor and add a dummy batch dim with unsqueeze -> (1, 3, 8, 8)
                    # Allocate input tensor array to GPU before running the forward pass
                    state_tensor = torch.tensor(state_matrix, dtype=torch.float32).unsqueeze(0).to(device)

                    need_critic = (ALGORITHM == "REINFORCE" and USE_BASELINE)
                    logits, state_value = model(state_tensor, need_critic=need_critic)


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
                    if reward == 1:
                        rollout_success = True

                    # Append values to trajectory
                    step_penalty = 0.0 if rollout_success else -0.005
                    log_probs.append(log_prob)
                    rewards.append(reward + step_penalty)
                    entropies.append(entropy)
                    if ALGORITHM == "REINFORCE" and USE_BASELINE:
                        state_values.append(state_value)

                    steps += 1
                
                group_log_probs.append(log_probs)
                group_rewards.append(sum(rewards)) 
                group_entropies.append(entropies)
                group_state_values.append(state_values)
                group_success.append(rollout_success)
                group_step_rewards.append(rewards)
                group_lengths.append(steps)
            
            policy_losses = []
            value_losses = []

            if ALGORITHM == "RLOO":
                R = np.array(group_rewards, dtype=np.float32) # turns into np array for contiguous efficiency
                group_advantages = np.zeros(CURRENT_GROUP_SIZE, dtype=np.float32)
                for i in range(CURRENT_GROUP_SIZE):
                    other_rewards = np.delete(R, i) # creates a copy of R but deletes i -> leaving one out
                    group_advantages[i] = R[i] - np.mean(other_rewards) # subtracts to use as baseline
                
                for i in range(CURRENT_GROUP_SIZE):
                    # or .mean() for length-normalized
                    lp_sum = torch.stack(group_log_probs[i]).sum()      
                    policy_losses.append(-group_advantages[i] * lp_sum)

                env_loss = torch.stack(policy_losses).sum() / CURRENT_GROUP_SIZE
            elif ALGORITHM == "GRPO":
                R = np.array(group_rewards, dtype=np.float32) 
                group_mean = np.mean(R)
                group_std = np.std(R) 
                
                group_advantages = np.zeros(CURRENT_GROUP_SIZE, dtype=np.float32)
                for i in range(CURRENT_GROUP_SIZE):
                    # apply z-score standardization + add a 1e-4 epsilon to prevent dividing by 0
                    group_advantages[i] = (R[i] - group_mean) / (group_std + 1e-4)
                
                for i in range(CURRENT_GROUP_SIZE):
                    adv = group_advantages[i]
                    for lp in group_log_probs[i]:
                        policy_losses.append(-lp * adv)
                
                env_loss = torch.stack(policy_losses).sum() / CURRENT_GROUP_SIZE
            elif ALGORITHM == "MaxRL":
                successes = np.array(group_success, dtype=np.float32)
                # count successful rollouts in group
                K = successes.sum()
                
                group_advantages = np.zeros(CURRENT_GROUP_SIZE, dtype=np.float32)
                for i in range(CURRENT_GROUP_SIZE):
                    if successes[i] > 0.0 and K > 0.0:
                        #  successes are scaled down inversely by how common success was in the group
                        group_advantages[i] = 1.0 / K
                    else:
                        # failed traj or batches carry an advantage of 0
                        group_advantages[i] = 0.0             
                
                for i in range(CURRENT_GROUP_SIZE):
                    adv = group_advantages[i]
                    for lp in group_log_probs[i]:
                        policy_losses.append(-lp * adv)
                
                env_loss = torch.stack(policy_losses).sum() / CURRENT_GROUP_SIZE
            elif ALGORITHM == "REINFORCE":
                rewards = group_rewards[0]   
                log_probs = group_log_probs[0]   
                state_values = group_state_values[0]

                discounted_returns = []
                G = 0
                step_rewards = group_step_rewards[0]
                # Calculate discounted reward by iterating in reverse
                for r in reversed(step_rewards):
                    # Horner's method-style G_t = r_t + gamma * G_{t+1}
                    G = r + GAMMA * G
                    # insert at 0 to reverse the reversed list to get discounted returns back in regular order (forward-temporal)
                    discounted_returns.insert(0, G)
            
                discounted_returns = torch.tensor(discounted_returns, dtype=torch.float32).to(device)
                
                if USE_BASELINE:
                    # Concatenate the list of single-item value tensors of dimension (1,1)
                    # into a continuous 1D tensor vector to align dimensionally with the discounted returns 
                    state_values = torch.cat(state_values).view(-1)
                    
                    # Zip collects together logprob and its corresponding return into tuples
                    for log_prob, G_t, V_s in zip(log_probs, discounted_returns, state_values):
                        advantage = G_t - V_s.item()
                        value_losses.append(F.smooth_l1_loss(V_s, G_t, beta=1.0))
                        policy_losses.append(-log_prob * advantage)
                else:
                    for log_prob, G_t in zip(log_probs, discounted_returns):
                        policy_losses.append(-log_prob * G_t)
                
                # Explanation of torch.stack()
                # CONVENIENCE: torch turns python list into 1D tensor of size (15) so we can call convenient .sum() function
                # IMPORTANCE: creates a master 'StackBackward' intersection node in the computational graph, allowing the trace backward:
                # loss -> sumbackward from sum -> stackbackward from stack -> mulbackward from log_prob * G_t -> categorical from dist
                # -> dist -> probs -> softmax -> logits -> model -> weights
                env_loss = torch.stack(policy_losses).sum()
                if USE_BASELINE:
                    # Sharing loss backprop for efficiency, learning the same representation of the maze
                    env_loss += CRITIC_COEFF * torch.stack(value_losses).sum()
            # 2D list comprehension, e (raw element, no expressions like 2*e) for traj in group entropies + for e in traj
            flat_entropies = [e for traj in group_entropies for e in traj]
            env_loss = env_loss - ENTROPY_COEFF * torch.stack(flat_entropies).mean()

            # gradient accumulation: /BATCH_SIZE so summed grads = MEAN grad over the batch
            (env_loss / BATCH_SIZE).backward()

            batch_success.append(float(np.mean(group_success)))
            batch_reward.append(float(np.mean(group_rewards)))
            batch_lengths.append(float(np.mean(group_lengths)))
            batch_entropy.append(torch.stack(flat_entropies).mean().item())
            if ALGORITHM == "REINFORCE" and USE_BASELINE:
                batch_c_loss += torch.stack(value_losses).sum().item()

        optimizer.step()
        global_step += 1

        episode_reward = float(np.mean(batch_reward)) # mean reward across batch
        mean_entropy = float(np.mean(batch_entropy))
        mean_steps = float(np.mean(batch_lengths))


        # Success requirements
        is_success = float(np.mean(batch_success))
        success_history.append(is_success)

        # Create rolling window
        rolling_window = success_history[-100:] # go from last 100 to end
        rolling_success_rate = np.mean(rolling_window) * 100

        if global_step % EVAL_INTERVAL == 0:
            print(f"\n--- Running Three-Metric Validation Suite Checkpoint at Step {global_step} ---")
            val_greedy_rate = evaluate(model, val_env, encode_as_2d_channels, num_mazes=NUM_VAL_MAZES, mode=EvalMode.GREEDY, max_steps=MAX_STEPS, modeltype="CNN", fixed_mazes=fixed_val_mazes)
            
            val_pass_k = evaluate(model, val_env, encode_as_2d_channels, num_mazes=NUM_VAL_MAZES, mode=EvalMode.PASS_K, N=10, max_steps=MAX_STEPS, modeltype="CNN", fixed_mazes=fixed_val_mazes)
            
            val_mean_1 = evaluate(model, val_env, encode_as_2d_channels, num_mazes=NUM_VAL_MAZES, mode=EvalMode.MEAN_K, N=1, max_steps=MAX_STEPS, modeltype="CNN", fixed_mazes=fixed_val_mazes)

            print(f"Validation Rates -> Greedy: {val_greedy_rate:.1f}% | Stochastic Pass@10: {val_pass_k:.1f}% | Stochastic Mean@1: {val_mean_1:.1f}%\n")

            wandb.log({
                "val_greedy_success_rate": val_greedy_rate,
                "val_stochastic_pass_10_rate": val_pass_k,
                "val_stochastic_mean_1_rate": val_mean_1,
                "global_step": global_step
            })

        if global_step % LOG_INTERVAL == 0:
            print(f"Update {global_step:04d} | Reward: {episode_reward:.2f} | Rolling Success: {rolling_success_rate:.1f}% | Steps: {mean_steps:.1f} | Entropy: {mean_entropy:.4f}")

            # Critic Loss
            c_loss = batch_c_loss if (ALGORITHM == "REINFORCE" and USE_BASELINE) else 0.0

            wandb.log({
                "mean_reward": episode_reward,
                "rolling_train_success_rate": rolling_success_rate,
                "response_length": mean_steps,
                "policy_entropy": mean_entropy,
                "critic_value_loss": c_loss,
                "global_step": global_step
            })
    
    torch.save(model.state_dict(),f"maze_{ALGORITHM}.pth")
    print(f"{ALGORITHM} training complete. Weights saved successfully!")

    wandb.finish()

if __name__ == "__main__":
    train_reinforce()        


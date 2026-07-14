import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import wandb

from environment import MazeEnv, VecMazeEnv
from maze_encodings import encode_as_2d_channels, encode_batch
from model import MazeCNN
from evaluate import evaluate, EvalMode
from maze_generation import build_fixed_eval_set

# use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using hardware accelerator: {device}")

# input size is fixed every step, so let cudnn autotune the conv algos once and reuse them
# (harmless on cpu, nice little speedup on the T4)
torch.backends.cudnn.benchmark = True

def resolve_path(filename):
    # prefer checkpoints/<name>, fall back to cwd (handy for flat Colab uploads)
    ckpt_path = os.path.join("checkpoints", filename)
    if os.path.exists(ckpt_path):
        return ckpt_path
    return filename

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
    

    ALGORITHM = "MaxRL"
    USE_BASELINE = False # Enable baseline critic value function

    USE_FIXED_VAL = True
    VAL_SEED = 12347
    NUM_VAL_MAZES = 50
    val_tag = f"valSeed{VAL_SEED}" if USE_FIXED_VAL else "valRandom"
    run_name = (
        f"RL_{ALGORITHM}_G{GROUP_SIZE}_{D}x{D} -{val_tag}"
        if ALGORITHM != "REINFORCE"
        else f"RL_{'REINFORCE_Baseline_CC:' + str(CRITIC_COEFF) + f"-{val_tag}" if USE_BASELINE else 'Vanilla_REINFORCE'}_{D}x{D} -{val_tag}"
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

    # Loads model to VRAM (checkpoints/BFS_BC_CNN-RL-starter.pth, or same folder on Colab)
    # Strict=false acknowledges that starter doesn't have values for fc_critic, etc. but that's okay
    starter_path = resolve_path("BFS_BC_CNN-RL-starter.pth")
    model.load_state_dict(torch.load(starter_path, map_location=device), strict=False)
    print(f"Loaded warm-start maze_CNN baseline from {starter_path}.")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    global_step = 0

    # Initialize success tracking window
    success_history = [] 

    print("Beginning on-policy RL optimization loop...")

    for update in range(TOTAL_UPDATES):
        model.train() # set to training mode so parameters track autograd updates
        CURRENT_GROUP_SIZE = GROUP_SIZE if ALGORITHM != "REINFORCE" else 1
        N = BATCH_SIZE * CURRENT_GROUP_SIZE # total parallel rollouts we run this update
        # critic head only needed for REINFORCE with baseline for backwards compatibility
        need_critic = (ALGORITHM == "REINFORCE" and USE_BASELINE)

        # sample BATCH_SIZE mazes; for each maze, duplicate the layout GROUP_SIZE times so consecutive
        # blocks of GROUP_SIZE rollouts share one maze. that shared block is one "group" for RLOO/GRPO/MaxRL.
        # (caches static maze for group rollouts + stores fixed start/goal coords of agent)
        mazes, starts, goals = [], [], []
        for _ in range(BATCH_SIZE):
            env.reset()
            for _ in range(CURRENT_GROUP_SIZE):
                mazes.append(np.copy(env.maze)) # restores/caches frozen maze obstacle layout
                starts.append(tuple(env.agent_pos)) # stores fixed start coords of agent
                goals.append(tuple(env.goal_pos))

        # one batched env that steps all N rollouts together -> one forward pass per timestep
        # (teleports every agent back to its start when constructed; steps reset to 0 for max step count)
        venv = VecMazeEnv(mazes, starts, goals, MAX_STEPS)

        # per-timestep buffers (stacks of step log-probs / entropies / rewards / values across the group)
        # later stacked into (T, N) once the rollout finishes
        logp_steps, ent_steps, rew_steps, mask_steps, value_steps = [], [], [], [], []
        reached = np.zeros(N, dtype=bool) # tracks success of each rollout (STOP on goal -> reward == 1)

        for t in range(MAX_STEPS):
            active = ~venv.done # boolean completion tracker: who's still alive at the start of this step
            if not active.any():
                break # everyone finished early, no point looping the rest of MAX_STEPS

            # (N, 3, D, D) snapshot of every current state
            # np array for C-style contiguous memory allocation, dtype float32 for casting bc that's what model uses
            states = encode_batch(venv.mazes, venv.agent, venv.goal)
            # Tensor object cast to track gradients and do fast matrix multiplication; allocate to GPU before the forward pass
            state_tensor = torch.from_numpy(states).to(device) # encode_batch already hands us float32

            # THE win: a single batched forward for all N rollouts instead of N tiny batch-1 ones
            logits, state_value = model(state_tensor, need_critic=need_critic)

            # create sampling distribution from our logits + sample from it
            # logits= lets torch do the softmax (same as softmax with dim=-1 since logits is (N, 5))
            # -> this is why we're using torch distribution btw, so we don't have to write logprob and entropy functions ourselves
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()

            # calculate logprob and policy entropy
            # this logprob = pi_theta(a_t | s_t)
            logp_steps.append(dist.log_prob(actions)) # (N,), kept on the graph for backprop
            ent_steps.append(dist.entropy())
            if need_critic:
                value_steps.append(state_value.view(-1)) # (N,) critic predictions, one per rollout

            # ONE gpu->cpu sync per timestep (not per env) -> then step the numpy env in bulk
            actions_np = actions.detach().cpu().numpy()
            reward_raw, done = venv.step(actions_np)
            # Tracks success of current rollout (env only gives reward==1 on STOP at goal)
            reached |= (reward_raw == 1.0)

            # Append values to trajectory: step penalty on every active, non-success step to discourage wandering
            # (same rule as before: no penalty once the +1 goal reward fired)
            penalty = np.where(active & (reward_raw != 1.0), -0.005, 0.0).astype(np.float32)
            wrong_stop = active & (actions_np == 4) & (reward_raw == 0.0)
            penalty = np.where(wrong_stop, -0.5, penalty)
            shaped = reward_raw + penalty

            rew_steps.append(torch.from_numpy(shaped).to(device)) # per-step negative reward shaping
            mask_steps.append(torch.from_numpy(active.astype(np.float32)).to(device)) # 1 while alive

        # stack the time buffers into (T, N)
        # Explanation of torch.stack()
        # CONVENIENCE: torch turns python list into one (T, N) tensor so we can call convenient .sum() functions
        # IMPORTANCE: creates a master 'StackBackward' intersection node in the computational graph, allowing the trace backward:
        # loss -> sumbackward from sum -> stackbackward from stack -> mulbackward from log_prob * adv -> categorical from dist
        # -> dist -> logits -> model -> weights
        logp = torch.stack(logp_steps) # with grad
        ent = torch.stack(ent_steps)   # with grad
        rew = torch.stack(rew_steps)   # shaped rewards, no grad (targets)
        mask = torch.stack(mask_steps) # 1.0 for steps a rollout was actually active

        # sum of step log-probabilities for each group rollout / total shaped reward / length
        sum_logp = (logp * mask).sum(dim=0)    # (N,)
        total_reward = (rew * mask).sum(dim=0) # (N,) accumulated terminal-ish total reward (sum of shaped steps)
        lengths = mask.sum(dim=0)              # (N,) response length in steps

        # exploration: mean policy entropy across every action we actually sampled (prevents premature mode collapse)
        entropy_bonus = (ent * mask).sum() / mask.sum().clamp(min=1.0)

        # reshape rewards to (BATCH_SIZE, GROUP) so each row is one maze's group of rollouts
        # turns into np array for contiguous efficiency + easy per-group (per-row) math
        R = total_reward.detach().cpu().numpy().reshape(BATCH_SIZE, CURRENT_GROUP_SIZE)

        value_term = torch.zeros((), device=device) # only REINFORCE with baseline fills this in

        if ALGORITHM == "RLOO":
            # leave-one-out: creates (effectively) a copy of each row but deletes i -> leaving one out,
            # then subtracts the mean of the others to use as baseline
            loo_mean = (R.sum(axis=1, keepdims=True) - R) / max(CURRENT_GROUP_SIZE - 1, 1)
            adv = (R - loo_mean).reshape(-1)
            adv_t = torch.from_numpy(adv.astype(np.float32)).to(device)
            # sum_logp is the per-rollout sum of log-probs (or .mean() for length-normalized)
            # since loss = -pi_theta(a_t | s_t) * A_i from the policy gradient theorem
            policy_loss = -(adv_t * sum_logp).mean()
        elif ALGORITHM == "GRPO":
            # apply z-score standardization within each group + add a 1e-4 epsilon to prevent dividing by 0
            g_mean = R.mean(axis=1, keepdims=True)
            g_std = R.std(axis=1, keepdims=True)
            adv = ((R - g_mean) / (g_std + 1e-4)).reshape(-1)
            adv_t = torch.from_numpy(adv.astype(np.float32)).to(device)
            policy_loss = -(adv_t * sum_logp).mean()
        elif ALGORITHM == "MaxRL":
            # count successful rollouts in each group (K per row) — for MaxRL tracking
            success_grid = reached.reshape(BATCH_SIZE, CURRENT_GROUP_SIZE).astype(np.float32)
            K = success_grid.sum(axis=1, keepdims=True)
            # successes are scaled down inversely by how common success was in the group (1/K);
            # failed traj or batches carry an advantage of 0
            adv = np.where((success_grid > 0) & (K > 0), 1.0 / np.maximum(K, 1.0), 0.0).reshape(-1)
            adv_t = torch.from_numpy(adv.astype(np.float32)).to(device)
            policy_loss = -(adv_t * sum_logp).mean()
        elif ALGORITHM == "REINFORCE":
            # Calculate discounted reward by iterating in reverse, for all N rollouts at once
            # Horner's method-style G_t = r_t + gamma * G_{t+1}
            # zero out post-done (masked) rewards first so G doesn't pick up junk if something leaks past done
            rew_masked = rew * mask
            returns = torch.zeros_like(rew_masked)
            G = torch.zeros(N, device=device)
            for t in range(rew_masked.size(0) - 1, -1, -1):
                G = rew_masked[t] + GAMMA * G
                returns[t] = G # builds discounted returns in forward-temporal order as we walk back

            if USE_BASELINE:
                # Concatenate / stack the list of per-step value tensors into a continuous (T, N)
                # tensor to align dimensionally with the discounted returns
                values = torch.stack(value_steps) # (T, N) with grad
                # Zip-style: each logprob with its corresponding return / value
                # .detach() so critic error doesn't leak into the policy gradient
                advantage = returns - values.detach()
                # huber (smooth L1) value loss — more robust under sparse/spiky returns than plain mse
                # summed over each rollout's real steps then averaged across rollouts
                v_loss = F.smooth_l1_loss(values, returns, beta=1.0, reduction="none")
                value_term = (v_loss * mask).sum(dim=0).mean()
            else:
                advantage = returns # vanilla REINFORCE just uses the raw discounted return G_t

            # since loss = -pi_theta(a_t | s_t) * G_t (or A_t) from policy gradient theorem
            policy_loss = -(logp * advantage * mask).sum(dim=0).mean()
        else:
            raise ValueError(f"Unknown ALGORITHM specified: {ALGORITHM}")

        # final loss = policy + (optional critic) - entropy bonus
        loss = policy_loss - ENTROPY_COEFF * entropy_bonus
        if need_critic:
            # Sharing loss backprop for efficiency, learning the same representation of the maze
            loss = loss + CRITIC_COEFF * value_term

        # whole batch is one graph now, so it's a single backward + step
        # (replaces the old gradient accumulation: /BATCH_SIZE so summed grads = MEAN grad over the batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1

        # --- metrics (all detached and cheap) ---
        episode_reward = total_reward.mean().item() # mean reward across batch
        mean_entropy = entropy_bonus.item()
        mean_steps = lengths.mean().item()
        batch_c_loss = value_term.item() if need_critic else 0.0

        # Success requirements — fraction of rollouts that actually reached the goal
        is_success = float(reached.mean())
        success_history.append(is_success)

        # Create rolling window
        rolling_window = success_history[-100:] # go from last 100 to end
        rolling_success_rate = np.mean(rolling_window) * 100

        if global_step % EVAL_INTERVAL == 0:
            print(f"\n--- Running Three-Metric Validation Suite Checkpoint at Step {global_step} ---")
            greedy_stats = evaluate(
                model, val_env, encode_as_2d_channels,
                num_mazes=NUM_VAL_MAZES, mode=EvalMode.GREEDY,
                max_steps=MAX_STEPS, modeltype="CNN",
                fixed_mazes=fixed_val_mazes, return_stats=True,
            )
            val_greedy_rate = greedy_stats["rate"]
            
            val_pass_k = evaluate(model, val_env, encode_as_2d_channels, num_mazes=NUM_VAL_MAZES, mode=EvalMode.PASS_K, N=10, max_steps=MAX_STEPS, modeltype="CNN", fixed_mazes=fixed_val_mazes)
            
            val_mean_1 = evaluate(model, val_env, encode_as_2d_channels, num_mazes=NUM_VAL_MAZES, mode=EvalMode.MEAN_K, N=1, max_steps=MAX_STEPS, modeltype="CNN", fixed_mazes=fixed_val_mazes)

            print(f"Validation Rates -> Greedy: {val_greedy_rate:.1f}% | Stochastic Pass@10: {val_pass_k:.1f}% | Stochastic Mean@1: {val_mean_1:.1f}%\n")

            wandb.log({
                "val_greedy_success_rate": val_greedy_rate,
                "val_greedy_timeout_frac": greedy_stats["timeout_frac"],
                "val_greedy_wrong_stop_frac": greedy_stats["wrong_stop_frac"],
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
    
    # save into checkpoints/ when that folder exists (local repo); otherwise cwd (Colab)
    out_name = f"maze_{ALGORITHM}.pth"
    out_dir = "checkpoints" if os.path.isdir("checkpoints") else "."
    out_path = os.path.join(out_dir, out_name)
    torch.save(model.state_dict(), out_path)
    print(f"{ALGORITHM} training complete. Weights saved to {out_path}!")

    wandb.finish()

if __name__ == "__main__":
    train_reinforce()        


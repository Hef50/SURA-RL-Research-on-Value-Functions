import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import wandb
import time

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

def set_seed(seed):
    # maze layouts come from np.random (generate_maze) and start/goal from random.sample
    # (place_start_goal), so a reproducible run has to pin both plus torch's sampling RNG
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_reinforce(
    # -- HYPERPARAMETERS (defaults; a sweep entry overrides any subset of these by name) --
    ALGORITHM="MaxRL",
    USE_BASELINE=False, # Enable baseline critic value function
    D=8, # size of maze
    GAMMA=0.99, # discount factor for future rewards
    LEARNING_RATE=5e-5, # Lower LR for RL for policy gradient stability
    TOTAL_UPDATES=200, # How many gradient steps (how many times optimizer.step() called)
    BATCH_SIZE=32, # Mazes (independent environments) per update
    GROUP_SIZE=8, # Rollouts per maze (1 for REINFORCE, __ for group methods)
    MAX_STEPS=60,
    LOG_INTERVAL=10, # Log interval to W&B -> every __ updates
    EVAL_INTERVAL=20, # Evaluate on held-out test mazes every __ updates
    CRITIC_COEFF=0.1, # downscaling critic's dominance to protect policy learning if needed
    ENTROPY_COEFF=0.01, # Exploration coefficient (beta) to scale the policy entropy bonus, preventing premature mode collapse
    USE_FIXED_VAL=True,
    VAL_SEED=12350,
    NUM_VAL_MAZES=100,
    USE_CRITIC=False, # replace the group statistic with a learned V(s_0) (the thesis knob)
    BINARY_REWARD=True, # advantages/critic targets use the raw success indicator, not shaped R
    VALUE_FLOOR=0.05, # 1/p explodes on near-unsolvable mazes -> caps the MaxRL weight at 20
    CRITIC_WARMUP=20, # updates spent fitting the critic before the policy trusts it
    SEED=0, # train-side RNG seed -> same config, different seed = a repeat, not a duplicate
    TAG="", # free-form suffix so one-off sweep variants are findable in W&B
    WANDB_GROUP=None, # collapses every run of a sweep into one group in the W&B UI
):
    set_seed(SEED)

    algo_config = ALGORITHM if ALGORITHM != "REINFORCE" else ("REINFORCE_Baseline" if USE_BASELINE else "Vanilla_REINFORCE")
    CURRENT_GROUP = GROUP_SIZE if ALGORITHM != "REINFORCE" else 1
    val_tag = f"valSeed{VAL_SEED}" if USE_FIXED_VAL else "valRandom"

    # one flat name builder for every algorithm: whatever can differ between two sweep runs
    # (algorithm, group size, lr, seed) has to be in the name, or W&B and the .pth files collide
    run_name = f"RL_{algo_config}_G{CURRENT_GROUP}_{D}x{D}_lr{LEARNING_RATE:g}_s{SEED}-{val_tag}"
    if USE_CRITIC:
        # "MaxRL" and "MaxRL + learned V" must never share a run name or a .pth file
        algo_config += "_V"
        run_name += "_V"
    if TAG:
        run_name += f"-{TAG}"

    # checkpoint filename mirrors the run name for the same reason: the old fixed
    # maze_{ALGORITHM}.pth meant run 2 silently overwrote run 1's weights
    out_name = f"maze_{algo_config}_G{CURRENT_GROUP}_{D}x{D}_s{SEED}" + (f"_{TAG}" if TAG else "") + ".pth"

    # Dependent Reference Counts
    TOTAL_ENVS = TOTAL_UPDATES * BATCH_SIZE # Num mazes
    TOTAL_ROLLOUTS = TOTAL_ENVS * CURRENT_GROUP

    wandb.init(
        project="SURA",
        name=run_name,
        group=WANDB_GROUP, # every run in a sweep shares this -> W&B can average/compare them
        reinit=True, # a sweep calls init() many times in one process
        config={
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "total_updates": TOTAL_UPDATES,
            "grid_size": D,
            "algorithm": algo_config,
            "group_size": CURRENT_GROUP,
            "gamma": GAMMA,
            "lr": LEARNING_RATE,
            "max_steps": MAX_STEPS,
            "use_baseline": USE_BASELINE,
            "critic_coeff": CRITIC_COEFF if (USE_CRITIC or USE_BASELINE) else 0.0,
            "use_critic": USE_CRITIC,
            "binary_reward": BINARY_REWARD,
            "value_floor": VALUE_FLOOR if USE_CRITIC else None,
            "critic_warmup": CRITIC_WARMUP if USE_CRITIC else 0,
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
        if update == 0:
            t_start = time.time()
        elif update == 20:
            per_update = (time.time() - t_start) / 20
            print(f"[timing] {per_update:.3f}s/update -> ~{per_update * TOTAL_UPDATES / 60:.1f} min for this run")
        model.train() # set to training mode so parameters track autograd updates
        CURRENT_GROUP_SIZE = GROUP_SIZE if ALGORITHM != "REINFORCE" else 1
        N = BATCH_SIZE * CURRENT_GROUP_SIZE # total parallel rollouts we run this update
        # critic head only needed for REINFORCE with baseline for backwards compatibility
        need_critic = USE_CRITIC or (ALGORITHM == "REINFORCE" and USE_BASELINE)

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

        # binary terminal reward: the only thing a verifier would give us in the LLM analogue
        r_bin = torch.from_numpy(reached.astype(np.float32)).to(device)  # (N,)
        adv_reward = r_bin if BINARY_REWARD else total_reward.detach()

        # reshape rewards to (BATCH_SIZE, GROUP) so each row is one maze's group of rollouts.
        # built from adv_reward so the group and critic paths are only ever different in the
        # baseline they use, never in what counts as reward
        R = adv_reward.cpu().numpy().reshape(BATCH_SIZE, CURRENT_GROUP_SIZE)

        p0_logit, p0 = None, None
        if USE_CRITIC:
            # every rollout in a group starts from the same state, so value_steps[0] holds one
            # copy of that group's V(s_0) per rollout -> no separate forward pass needed
            p0_logit = value_steps[0]                 # (N,) with grad
            p0 = torch.sigmoid(p0_logit).detach()     # normalizers must not backprop into the critic
        critic_ready = USE_CRITIC and global_step >= CRITIC_WARMUP

        value_term = torch.zeros((), device=device) 

        if ALGORITHM == "RLOO":
            if critic_ready:
                # the leave-one-out mean was only ever an estimate of E[R | s_0].
                # V(s_0) estimates the same quantity from one rollout instead of G-1 others
                adv_t = adv_reward - p0
                policy_loss = -(adv_t * sum_logp).mean()
            else:
                # leave-one-out: creates (effectively) a copy of each row but deletes i -> leaving one out,
                # then subtracts the mean of the others to use as baseline
                loo_mean = (R.sum(axis=1, keepdims=True) - R) / max(CURRENT_GROUP_SIZE - 1, 1)
                adv = (R - loo_mean).reshape(-1)
                adv_t = torch.from_numpy(adv.astype(np.float32)).to(device)
                # sum_logp is the per-rollout sum of log-probs (or .mean() for length-normalized)
                # since loss = -pi_theta(a_t | s_t) * A_i from the policy gradient theorem
                policy_loss = -(adv_t * sum_logp).mean()
        elif ALGORITHM == "GRPO":
            if critic_ready:
                # binary reward => Var[R | s_0] = p(1-p), so the critic's mean pins the normalizer too
                # and GRPO needs no second-moment head
                std = torch.sqrt(torch.clamp(p0 * (1.0 - p0), min=1e-4))
                adv_t = (adv_reward - p0) / (std + 1e-4)
                policy_loss = -(adv_t * sum_logp).mean()
            else:
                # apply z-score standardization within each group + add a 1e-4 epsilon to prevent dividing by 0
                g_mean = R.mean(axis=1, keepdims=True)
                g_std = R.std(axis=1, keepdims=True)
                adv = ((R - g_mean) / (g_std + 1e-4)).reshape(-1)
                adv_t = torch.from_numpy(adv.astype(np.float32)).to(device)
                policy_loss = -(adv_t * sum_logp).mean()
        elif ALGORITHM == "MaxRL":
            if critic_ready:
                # 1/K was a plug-in estimate of 1/p with p ≈ K/G, so G*V(s_0) substitutes for K.
                # the G factor keeps the advantage scale matched to the group runs — without it the
                # effective learning rate jumps 8x and the comparison is confounded
                p_floor = p0.clamp(min=VALUE_FLOOR)
                adv_t = adv_reward / (CURRENT_GROUP_SIZE * p_floor)
                policy_loss = -(adv_t * sum_logp).sum() / BATCH_SIZE
            else:
                # count successful rollouts in each group (K per row) — for MaxRL tracking
                success_grid = reached.reshape(BATCH_SIZE, CURRENT_GROUP_SIZE).astype(np.float32)
                K = success_grid.sum(axis=1, keepdims=True)
                # successes are scaled down inversely by how common success was in the group (1/K);
                # failed traj or batches carry an advantage of 0
                adv = np.where((success_grid > 0) & (K > 0), 1.0 / np.maximum(K, 1.0), 0.0).reshape(-1)
                adv_t = torch.from_numpy(adv.astype(np.float32)).to(device)
                policy_loss = -(adv_t * sum_logp).sum() / BATCH_SIZE
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
        if USE_CRITIC:
            # target = did THIS rollout succeed. all G rollouts of a maze share s_0, so BCE over
            # the batch regresses V(s_0) onto the group's realized K/G — the very statistic the
            # group methods obtain by brute force
            value_term = F.binary_cross_entropy_with_logits(p0_logit, r_bin)
        # final loss = policy + (optional critic) - entropy bonus
        loss = policy_loss - ENTROPY_COEFF * entropy_bonus
        if USE_CRITIC and not critic_ready:
            # policy sits still while the critic learns what a solvable maze looks like.
            # unscaled because CRITIC_COEFF exists to stop the critic drowning out the policy,
            # which is moot while the policy isn't learning
            loss = value_term
        elif need_critic:
            # Sharing loss backprop for efficiency, learning the same representation of the maze
            loss = loss + CRITIC_COEFF * value_term

        # whole batch is one graph now, so it's a single backward + step
        # (replaces the old gradient accumulation: /BATCH_SIZE so summed grads = MEAN grad over the batch)
        optimizer.zero_grad()
        loss.backward()
        if USE_CRITIC and not critic_ready:
            # critic-only phase: keep value gradients out of the BC-warm-started shared trunk
            for pname, p in model.named_parameters():
                if not pname.startswith(("fc_critic", "value_head")) and p.grad is not None:
                    p.grad = None

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

        # is the critic predicting THIS maze's difficulty, or just the global success rate?
        # (most meaningful at G>1, where k_grp is a real rate rather than a single 0/1 draw)
        critic_mae, critic_corr, floor_frac = 0.0, 0.0, 0.0
        if USE_CRITIC:
            p_grp = p0.view(BATCH_SIZE, CURRENT_GROUP_SIZE)[:, 0].cpu().numpy()
            k_grp = reached.reshape(BATCH_SIZE, CURRENT_GROUP_SIZE).mean(axis=1)
            critic_mae = float(np.abs(p_grp - k_grp).mean())
            critic_corr = float(np.corrcoef(p_grp, k_grp)[0, 1]) if p_grp.std() > 1e-6 else 0.0
            floor_frac = float((p0 <= VALUE_FLOOR).float().mean())

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
            
            val_mean_1 = evaluate(model, val_env, encode_as_2d_channels, num_mazes=NUM_VAL_MAZES, mode=EvalMode.MEAN_K, N=16, max_steps=MAX_STEPS, modeltype="CNN", fixed_mazes=fixed_val_mazes)

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
            if USE_CRITIC:
                print(f"   critic -> BCE: {batch_c_loss:.4f} | MAE vs K/G: {critic_mae:.3f} | corr: {critic_corr:+.3f} | floored: {floor_frac:.1%}")
            # Critic Loss — now covers USE_CRITIC runs, not just REINFORCE + baseline
            c_loss = batch_c_loss if need_critic else 0.0
            wandb.log({
                "mean_reward": episode_reward,
                "rolling_train_success_rate": rolling_success_rate,
                "response_length": mean_steps,
                "policy_entropy": mean_entropy,
                "critic_value_loss": c_loss,
                "critic_mae_vs_group": critic_mae,
                "critic_corr_vs_group": critic_corr,
                "critic_floor_frac": floor_frac,
                "global_step": global_step
            })

    
    # save into checkpoints/ when that folder exists (local repo); otherwise cwd (Colab)
    # (out_name was built alongside run_name so the two always agree)
    out_dir = "checkpoints" if os.path.isdir("checkpoints") else "."
    out_path = os.path.join(out_dir, out_name)
    torch.save(model.state_dict(), out_path)
    print(f"{run_name} complete. Weights saved to {out_path}!")

    wandb.finish()
    # handed back so the sweep driver can record which checkpoint belongs to which config
    return run_name, out_path

if __name__ == "__main__":
    SWEEP = "longrun_v1"
    SEEDS = [0]  # one seed first 
    COMMON = {"TOTAL_UPDATES": 2000, "EVAL_INTERVAL": 50}
    RUNS = [
        # the headline pair - this is what the acceptance criterion is about
        {"ALGORITHM": "RLOO",  "GROUP_SIZE": 8},
        {"ALGORITHM": "MaxRL", "GROUP_SIZE": 8},
        {"ALGORITHM": "RLOO",  "GROUP_SIZE": 32, "BATCH_SIZE": 8, "TAG": "G32"},
        {"ALGORITHM": "MaxRL", "GROUP_SIZE": 32, "BATCH_SIZE": 8, "TAG": "G32"},
    ]

    out_dir = "checkpoints" if os.path.isdir("checkpoints") else "."
    manifest_path = os.path.join(out_dir, f"runs_{SWEEP}.json")

    # resume - a Colab disconnect must not cost the runs that already finished
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else []
    def cfg_key(c):
        return json.dumps({k: v for k, v in c.items()
                           if k not in ("run_name", "checkpoint")}, sort_keys=True)
    done = {cfg_key(m) for m in manifest}

    for seed in SEEDS:
        for run in RUNS:
            cfg = {"WANDB_GROUP": SWEEP, "SEED": seed, **COMMON, **run}
            if cfg_key(cfg) in done:
                print(f"== skipping already-finished run: {cfg}")
                continue
            print(f"\n===== sweep {SWEEP} | run {len(manifest) + 1} | {cfg} =====")
            try:
                name, path = train_reinforce(**cfg)
            except Exception as err:
                # one bad config shouldn't cost you the other runs overnight
                print(f"!! run failed: {type(err).__name__}: {err} -- skipping to next config")
                wandb.finish(exit_code=1)
                continue
            manifest.append({"run_name": name, "checkpoint": path, **cfg})
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

    print(f"\nSweep done: {len(manifest)} runs recorded -> {manifest_path}")


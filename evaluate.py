import torch
import torch.nn.functional as F
import random
import numpy as np
from enum import Enum

from environment import MazeEnv, VecMazeEnv
from maze_encodings import encode_as_channels, encode_batch
from model import MazeMLP

class EvalMode(Enum):
    GREEDY = "greedy"
    PASS_K = "pass_k"
    MEAN_K = "mean_k"

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

def evaluate(model, env, encoding_fn, num_mazes=100, mode=EvalMode.GREEDY,
             N=10, max_steps=50, modeltype="CNN", fixed_mazes=None, return_stats=False):
    # Unified Evaluation Function to evaluate deterministic (Greedy) and stochastic (Pass@k, Mean@k) rollouts cleanly
    # vectorized: every rollout (across all mazes + attempts) runs as one big parallel batch through a
    # single forward pass per timestep, instead of one batch-1 forward per step per rollout
    model.eval()

    # Each pytorch tensor has a .device() method, so we're going to
    # the first tensor of the model's parameters (NEXT to the model header) to find its device
    device = next(model.parameters()).device

    # Greedy only requires a single deterministic attempt per maze; stochastic modes get N sampled attempts
    rollouts_per_maze = 1 if mode == EvalMode.GREEDY else N

    # if fixed: ignore num_mazes, use len(fixed_mazes)
    n = len(fixed_mazes) if fixed_mazes is not None else num_mazes

    # gather the maze set once (either the fixed held-out set or freshly generated ones)
    base_mazes, base_starts, base_goals = [], [], []
    for i in range(n):
        if fixed_mazes is not None:
            item = fixed_mazes[i]
            base_mazes.append(np.copy(item["maze"]))
            # Freezes coordinates into immutable tuples to prevent aliasing bug
            base_starts.append(tuple(item["start_pos"]))
            base_goals.append(tuple(item["goal_pos"]))
        else:
            env.reset()
            # If env.maze is ever variable, this freezes maze so successive env.reset()s don't create new mazes
            base_mazes.append(np.copy(env.maze))
            base_starts.append(tuple(env.agent_pos))
            base_goals.append(tuple(env.goal_pos))

    # replicate each maze rollouts_per_maze times so all attempts of all mazes run together as one batch
    # (same freeze/restore idea as the old per-maze loop, just bulked out beforehand)
    mazes, starts, goals = [], [], []
    for i in range(n):
        for _ in range(rollouts_per_maze):
            mazes.append(base_mazes[i])
            starts.append(base_starts[i])
            goals.append(base_goals[i])

    venv = VecMazeEnv(mazes, starts, goals, max_steps)
    M = venv.N # total parallel rollouts = n * rollouts_per_maze

    # Tracks: complete solves, wrong STOPs, timeouts (for greedy diagnostics / return_stats)
    # success = STOP on goal (reward == 1) — same definition as training / MaxRL
    # wrong_stop = issued STOP off-goal; timeout = ran out the clock without solving
    reached = np.zeros(M, dtype=bool)
    wrong_stop = np.zeros(M, dtype=bool)

    with torch.no_grad():
        for _ in range(max_steps):
            active = ~venv.done # who's still running at the start of this step
            if not active.any():
                break # everyone finished, no point looping the rest of max_steps

            # [] for (3D^2) -> batched: encode_batch builds (M, 3, D, D) contiguous float32,
            # then Tensor cast for fast matrix multiplication / model input
            states = encode_batch(venv.mazes, venv.agent, venv.goal) # (M, 3, D, D)
            if modeltype == "MLP":
                # flatten channels the same way encode_as_channels does (walls | agent | goal)
                state_tensor = torch.from_numpy(states.reshape(M, -1)).to(device)
                logits = model(state_tensor)
            elif modeltype == "CNN":
                # encoding_fn is the single-maze encoder; we use encode_batch (its vectorized cousin) here
                # skip critic head — eval only needs the policy logits
                state_tensor = torch.from_numpy(states).to(device)
                logits, _ = model(state_tensor, need_critic=False)
            else:
                raise ValueError(f"Unknown MODEL_TYPE specified: {modeltype}")

            if mode == EvalMode.GREEDY:
                actions = torch.argmax(logits, dim=1) # deterministic argmax, dim=1 selects actions
            else:
                # Convert logits to discrete probability distributions, dim=1 to select actions, not batches (dim=0)
                probs = F.softmax(logits, dim=1)
                # Randomly sample an index according to probability array distribution
                actions = torch.multinomial(probs, num_samples=1).squeeze(1)

            # ONE sync per timestep for the whole batch (same win as training)
            actions_np = actions.detach().cpu().numpy()

            # a STOP (action 4) issued off the goal is a "wrong stop"
            # classify using pre-step positions so we catch it before the env marks done
            on_goal = np.all(venv.agent == venv.goal, axis=1)
            wrong_stop |= active & (actions_np == 4) & (~on_goal)

            reward_raw, _ = venv.step(actions_np)
            # env only gives reward==1 on STOP at goal — keep this aligned with train_reinforce / MaxRL
            reached |= (reward_raw == 1.0)

        # reshape per-rollout outcomes to (n, rollouts_per_maze) so we can aggregate per maze
        solved_grid = reached.reshape(n, rollouts_per_maze)

        if mode == EvalMode.MEAN_K:
            # fractional rollout success per maze (successful_attempts / rollouts_per_maze), then averaged
            rate = float(solved_grid.mean(axis=1).mean()) * 100
        else:
            # Greedy (1 attempt) and Pass@k both count a maze as solved if ANY attempt reached the goal
            # (Pass@k no longer early-breaks, but the "at least one success" math is identical —
            #  we just pay for the unused attempts so the whole batch can stay vectorized)
            rate = float(solved_grid.any(axis=1).mean()) * 100

    if return_stats:
        denom = max(M, 1)
        # a rollout that neither solved nor wrong-stopped simply ran out the clock -> timeout / never emitted STOP
        timeout = ~reached & ~wrong_stop
        return {
            "rate": rate,
            "timeout_frac": float(timeout.sum()) / denom,     # loop / never emitted STOP
            "wrong_stop_frac": float(wrong_stop.sum()) / denom,
            "solved_frac": float(reached.sum()) / denom,
        }
    return rate


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
        
        # greedy_rate = evaluate_model_policy_greedy(model, env, encode_as_channels, num_mazes=100, max_steps=MAX_STEPS)
        # stoch_rate = evaluate_stochastic_pass_k(model, env, encode_as_channels, num_mazes=100, N=10, max_steps=MAX_STEPS)
        
        # print(f"Greedy Policy Success Rate: {greedy_rate:.1f}%")
        # print(f"Stochastic Pass@10 Success Rate: {stoch_rate:.1f}%")
    except FileNotFoundError:
        print("Could not find weights file. Run train.py first to create maze_mlp.pth.")

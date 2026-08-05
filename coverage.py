import os
import json

import numpy as np
import torch
import matplotlib.pyplot as plt

from environment import MazeEnv
from maze_encodings import encode_as_2d_channels
from model import MazeCNN
from evaluate import evaluate, EvalMode
from maze_generation import build_fixed_eval_set

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_path(filename):
    # same convention as train-reinforce.py: prefer checkpoints/<name>, fall back to cwd (flat Colab uploads)
    ckpt_path = os.path.join("checkpoints", filename)
    return ckpt_path if os.path.exists(ckpt_path) else filename


def pass_at_k(n, c, k):
    # unbiased pass@k estimator (Chen et al. 2021, HumanEval): 1 - C(n-c, k) / C(n, k)
    # = the chance a random k-subset of our n sampled rollouts contains at least one success.
    # computed as a running product instead of factorials so big n never overflows
    if n - c < k:
        return 1.0  # fewer than k failures exist, so every k-subset must contain a success
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def success_counts(model, eval_mazes, D, max_steps, n_samples, chunk):
    # draw n_samples stochastic rollouts per maze and count how many solved each maze.
    # chunked because evaluate() puts len(eval_mazes) * chunk rollouts in ONE forward pass
    # 200 mazes x 256 samples at once would be 51k rollouts of conv activations in VRAM
    env = MazeEnv(D=D, max_steps=max_steps)  # unused when fixed_mazes is passed, but evaluate() wants it
    counts = np.zeros(len(eval_mazes), dtype=np.int64)
    drawn = 0
    while drawn < n_samples:
        this_chunk = min(chunk, n_samples - drawn)
        grid = evaluate(
            model, env, encode_as_2d_channels,
            mode=EvalMode.PASS_K, N=this_chunk, max_steps=max_steps,
            modeltype="CNN", fixed_mazes=eval_mazes, return_grid=True,
        )
        counts += grid.sum(axis=1)  # independent samples, so chunks just add
        drawn += this_chunk
        print(f"  sampled {drawn}/{n_samples} rollouts per maze", end="\r")
    print()
    return counts, drawn


if __name__ == "__main__":
    D = 8
    MAX_STEPS = 60
    HIDDEN_DIM = 128

    # held-out test set: a different seed from the VAL_SEED training tunes against,
    # so coverage isn't measured on mazes any run was checkpoint-selected on
    TEST_SEED = 999
    NUM_TEST_MAZES = 200

    K_VALUES = [1, 10, 32, 64, 128]
    N_SAMPLES = 256  # >= 2 * max(K_VALUES) so pass@128 averages over many subsets, not one draw
    CHUNK = 32

    # label -> checkpoint filename. "Base" is the BC warm-start, i.e. the model RL started from
    CHECKPOINTS = {
        "Base (BC)": "BFS_BC_CNN-RL-starter.pth",
        "MaxRL": "maze_MaxRL_G8_8x8_s0.pth",
        "GRPO": "maze_GRPO_G8_8x8_s0.pth",
        "RLOO": "maze_RLOO_G8_8x8_s0.pth",
    }

    test_mazes = build_fixed_eval_set(D=D, num_mazes=NUM_TEST_MAZES, seed=TEST_SEED)
    print(f"Coverage sweep: {NUM_TEST_MAZES} held-out {D}x{D} mazes, n={N_SAMPLES} samples each")

    curves = {}
    for label, filename in CHECKPOINTS.items():
        path = resolve_path(filename)
        if not os.path.exists(path):
            print(f"skipping {label}: {path} not found")
            continue

        model = MazeCNN(d=D, hidden_dim=HIDDEN_DIM).to(device)
        # strict=False: the BC starter has no critic-head weights
        model.load_state_dict(torch.load(path, map_location=device), strict=False)
        print(f"{label} <- {path}")

        counts, n = success_counts(model, test_mazes, D, MAX_STEPS, N_SAMPLES, CHUNK)
        curves[label] = [100.0 * float(np.mean([pass_at_k(n, c, k) for c in counts])) for k in K_VALUES]
        print("  " + " | ".join(f"pass@{k}: {v:.1f}%" for k, v in zip(K_VALUES, curves[label])))

    os.makedirs("assets", exist_ok=True)
    with open(os.path.join("assets", "coverage_pass_at_k.json"), "w") as f:
        json.dump({"k_values": K_VALUES, "n_samples": N_SAMPLES, "curves": curves}, f, indent=2)

    plt.figure(figsize=(5.5, 4.5))
    for label, ys in curves.items():
        plt.plot(K_VALUES, ys, marker="^", linewidth=2, markersize=6, label=label)
    plt.xscale("log", base=2)  # the paper's x-axis is log2 k; a crossover is only visible this way
    plt.xticks(K_VALUES, [str(k) for k in K_VALUES])
    plt.xlabel("Number of samples $k$")
    plt.ylabel("Coverage (pass@$k$) %")
    plt.title(f"{D}x{D} mazes, {NUM_TEST_MAZES} held-out, n={N_SAMPLES}")
    plt.ylim(0, 100)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join("assets", "coverage_pass_at_k.png"), dpi=200, bbox_inches="tight")
    print("Saved assets/coverage_pass_at_k.png")
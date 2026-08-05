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

    K_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]
    N_SAMPLES = 256  # >= 2 * max(K_VALUES) so pass@128 averages over many subsets, not one draw
    CHUNK = 32

    # label -> checkpoint filename, built from the sweep manifest so nothing is hand-typed
    SWEEP = "valuefn_v1"
    with open(resolve_path(f"runs_{SWEEP}.json")) as f:
        runs = [r for r in json.load(f) if r["SEED"] == 0]
    def label_of(r):
        name = r["ALGORITHM"] + (" + V" if r.get("USE_CRITIC") else "")
        name += f" (G={r.get('GROUP_SIZE', 8)})"
        return name + (f" [{r['TAG']}]" if r.get("TAG") else "")
    CHECKPOINTS = {"Base (BC)": "BFS_BC_CNN-RL-starter.pth"}
    for r in runs:
        CHECKPOINTS[label_of(r)] = os.path.basename(r["checkpoint"])
    test_mazes = build_fixed_eval_set(D=D, num_mazes=NUM_TEST_MAZES, seed=TEST_SEED)
    print(f"Coverage sweep: {NUM_TEST_MAZES} held-out {D}x{D} mazes, n={N_SAMPLES} samples each")
    # sample ONCE per checkpoint and keep the raw per-maze counts. every k value, every re-plot
    # and the calibration figure all come out of these without touching the GPU again
    all_counts = {}
    for label, filename in CHECKPOINTS.items():
        path = resolve_path(filename)
        if not os.path.exists(path):
            print(f"skipping {label}: {path} not found")
            continue
        model = MazeCNN(d=D, hidden_dim=HIDDEN_DIM).to(device)
        model.load_state_dict(torch.load(path, map_location=device), strict=False)
        print(f"{label} <- {path}")
        counts, n = success_counts(model, test_mazes, D, MAX_STEPS, N_SAMPLES, CHUNK)
        all_counts[label] = counts.tolist()
    curves = {lbl: [100.0 * float(np.mean([pass_at_k(N_SAMPLES, c, k) for c in cts])) for k in K_VALUES]
            for lbl, cts in all_counts.items()}
    for lbl, ys in curves.items():
        print(f"{lbl:32s} " + " | ".join(f"@{k}: {v:.1f}" for k, v in zip(K_VALUES, ys)))
    os.makedirs("assets", exist_ok=True)
    with open(os.path.join("assets", f"coverage_{SWEEP}.json"), "w") as f:
        json.dump({"k_values": K_VALUES, "n_samples": N_SAMPLES,
                "counts": all_counts, "curves": curves}, f, indent=2)
    # one figure per algorithm — twelve curves on a single axis is unreadable
    for algo in ["MaxRL", "GRPO", "RLOO"]:
        subset = {l: v for l, v in curves.items() if l.startswith(algo) or l.startswith("Base")}
        if len(subset) <= 1:
            continue
        plt.figure(figsize=(5.5, 4.5))
        for label, ys in subset.items():
            plt.plot(K_VALUES, ys, marker="^", linewidth=2, markersize=6, label=label)
        plt.xscale("log", base=2)
        plt.xticks(K_VALUES, [str(k) for k in K_VALUES])
        plt.xlabel("Number of samples $k$")
        plt.ylabel("Coverage (pass@$k$) %")
        plt.title(f"{algo}: {D}x{D}, {NUM_TEST_MAZES} held-out, n={N_SAMPLES}")
        plt.ylim(0, 100)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8)
        plt.savefig(os.path.join("assets", f"coverage_{algo}.png"), dpi=200, bbox_inches="tight")
        print(f"Saved assets/coverage_{algo}.png")
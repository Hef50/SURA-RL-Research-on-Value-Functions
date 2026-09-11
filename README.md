# SURA Research Project: Investigating Value Functions in Terminal-Reward Reinforcement Learning

A research project investigating **whether learned value functions provide meaningful utility in terminal-reward (sparse-reward) reinforcement learning** — the regime that underlies modern RL for large language models (LLMs) — studied efficiently in gridworld maze environments.

> **Status:** Active research (Summer Undergraduate Research Apprenticeship, June–July 2026; continuing into the academic year). Work in progress.
> Conducted in [Prof. Andrea Zanette's lab](https://azanette.com/) at Carnegie Mellon University, advised day-to-day by PhD student Daman Arora.

---

## Overview

Modern reinforcement learning for reasoning LLMs is dominated by **terminal-reward** methods: a model generates a full response and receives a single scalar reward at the end (e.g., 1 if the answer is correct, 0 otherwise). Increasingly, the field has moved toward **critic-free, group-based** optimization methods (GRPO, RLOO, and MaxRL) that estimate a baseline from a *group* of sampled rollouts rather than from a learned value function.

Value functions are a classical tool for reducing variance and improving credit assignment in RL, and they are well-established in dense-reward settings. Yet their utility in the terminal-reward regime remains **an open and actively debated question**: the field largely moved away from critics without a clean, controlled study of whether they help here.

This project builds a low-compute maze testbed to study that question directly, and to explore a specific hypothesis:

> **Group-based methods like MaxRL currently require multiple rollouts per prompt (a "group") to estimate their learning signal, which is computationally expensive. A learned value function could, in principle, provide that signal from a single rollout, potentially making MaxRL-style training substantially cheaper.**

The connection is concrete. For a binary terminal reward, the value of a state under the current policy *is* the probability of success from that state. So for a group of `G` rollouts from a start state `s₀`, the expected number of successes is `E[K] = G · V(s₀)` — which is exactly the statistic the group methods buy through brute-force sampling. A critic that predicts it directly could replace the group.

---

## Research Questions

1. **Do learned value-function baselines/critics measurably improve performance** (sample efficiency, success rate, training stability) over critic-free policy-gradient methods in a controlled terminal-reward setting?
2. **How does any benefit depend on task difficulty, episode horizon, and reward sparsity?**
3. *(Exploratory)* **Can a value function substitute for group sampling in MaxRL-style training**, enabling single-rollout updates that retain MaxRL's advantages at lower cost?

A known conceptual risk, stated up front: MaxRL normalizes by `K`, the number of successful rollouts in a group, which breaks the standard unbiased-baseline argument. Whether a per-step value signal is even well-posed under that normalization is itself an open question, and answering it in the negative would still be a legitimate result.

---

## Approach

The project uses **procedurally generated maze environments** as a cheap, controlled proxy for terminal-reward RL. Mazes are small enough to run many controlled comparisons on a CPU or a free Colab GPU, while preserving the terminal-reward structure that characterizes LLM RL.

### Environment

An agent navigates a randomly generated `D×D` maze (built with randomized Prim's algorithm) toward a goal, choosing from five actions `{up, down, left, right, stop}`. The reward is strictly terminal and binary: **+1 only if the agent issues `stop` while on the goal cell, and 0 otherwise** — including a premature `stop`, which ends the episode immediately. Walking onto the goal is not enough; the agent must recognize it has arrived. Bumping a wall or the boundary leaves the agent in place but still consumes a step. Start and goal positions are randomized per maze.

The default setting is `D=8` with a 60-step horizon.

### Observation

The policy sees four `D×D` channels:

| Channel | Contents |
|---|---|
| 0 | walls (1 = wall, 0 = open) |
| 1 | one-hot agent position |
| 2 | one-hot goal position |
| 3 | constant plane of `t / max_steps` |

The timestep is included because with a step limit the true value of a state genuinely depends on how many steps remain — a time-blind critic is fitting a target that isn't well-defined on its inputs. It also makes the policy non-stationary in time, which matters for a deterministic environment: a purely positional greedy policy either solves a maze or cycles forever.

Manhattan distance is deliberately **not** provided. It is inferable from the agent and goal channels, and handing it over would make the value function trivial to learn — the point is to give the policy the simplest complete description of the state and have it learn the value function itself.

### Methods

All algorithms are implemented from scratch in PyTorch, sharing one training loop:

- **Behavior Cloning (BC)** — supervised imitation of a BFS shortest-path expert; also the RL warm-start, mirroring SFT-then-RL in LLM training.
- **REINFORCE** — vanilla policy gradient.
- **REINFORCE with a value-function baseline** — the classical variance-reduction approach.
- **RLOO** (REINFORCE Leave-One-Out) — group baseline from the other samples in the group.
- **GRPO** (Group Relative Policy Optimization) — group-normalized advantages.
- **MaxRL** — maximum-likelihood, success-normalized: only successful rollouts contribute, each weighted `1/K`.

Each group method also has a **learned-critic variant** (`USE_CRITIC=True`) in which `V(s₀)` replaces the group statistic. The critic is a second head on the shared convolutional trunk, trained by binary cross-entropy against each rollout's realized success. Because every rollout in a group starts from the same `s₀`, the BCE minimizer for a group is exactly `K/G`.

### Evaluation

Policies are compared on three metrics, all computed by one mode-switched function in `evaluate.py`:

- **greedy** — argmax action at every step, one rollout per maze.
- **mean@1** — expected success of a single *sampled* rollout. Estimated with `N` samples per maze and averaged; the estimand is the same for every `N`, only the variance changes.
- **pass@k** — a maze counts as solved if any of `k` sampled rollouts succeeds. `coverage.py` uses the unbiased Chen et al. (2021) estimator rather than a raw any-of-`k` indicator.

Training additionally logs pass@1 and pass@`G` measured directly on the training batch, which is free: at `GROUP_SIZE=32` every group already *is* 32 samples of one maze.

All experiments are tracked with [Weights & Biases](https://wandb.ai/) (project `SURA`). **Logging is step-based, never epoch-based** — an epoch contains a different number of gradient steps at different dataset sizes, which makes curves incomparable across runs.

---

## Repository Structure

```
.
├── train-reinforce.py          # RL training + the sweep driver (main entry point)
├── train_behavior_cloning.py   # BC warm-start training
├── coverage.py                 # post-hoc pass@k coverage curves
├── bfs_expert.py               # BFS shortest-path expert for BC
├── environment.py              # MazeEnv + VecMazeEnv (batched rollouts)
├── maze_generation.py          # Prim mazes + build_fixed_eval_set
├── maze_encodings.py           # single-maze + batched encodings
├── model.py                    # MazeCNN / MazeMLP (shared trunk, actor + critic heads)
├── evaluate.py                 # unified greedy / pass@k / mean@k evaluation
├── requirements.txt
├── checkpoints/                # .pth weights + sweep manifests
├── assets/                     # figures
├── archive/                    # superseded one-off scripts (do not use)
└── README.md
```

`VecMazeEnv` is what training and evaluation actually run: a batched NumPy environment that steps `N` rollouts together, so each timestep costs one forward pass instead of `N`.

---

## Setup

```bash
git clone https://github.com/Hef50/SURA.git
cd SURA

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

Developed and pinned against **Python 3.12**.

**PyTorch.** Install the build matching your platform first, via [pytorch.org](https://pytorch.org/get-started/locally/). On Windows a plain `pip install torch` gives you a CPU-only build with no warning, and RL training will be unusably slow.

**Weights & Biases.** Every training entry point calls `wandb.init()` unconditionally, so you need either an account:

```bash
wandb login
```

or the offline mode, which works with no account at all:

```bash
# Windows PowerShell
$env:WANDB_MODE = "offline"
# macOS/Linux
export WANDB_MODE=offline
```

---

## Configuration

**There is no command-line interface** — no `argparse`, no config files, no environment variables. Every knob is a constant or a keyword-argument default edited in-file:

| What | Where |
|---|---|
| RL hyperparameters | the `train_reinforce(...)` signature in `train-reinforce.py` |
| Which runs a sweep executes | the `RUNS` and `COMMON` blocks in `train-reinforce.py`'s `__main__` |
| BC hyperparameters | the top of `train_behavioral_cloning()` in `train_behavior_cloning.py` |
| pass@k evaluation | the constants block in `coverage.py`'s `__main__` |

A sweep entry overrides any subset of `train_reinforce`'s defaults by name, so `{"ALGORITHM": "GRPO", "GROUP_SIZE": 32}` is a complete run specification and everything else falls back to the signature default.

---

## Usage

The pipeline runs in four steps. Steps 1 and 2 produce the warm-start that step 3 depends on.

### 0. Verify the install

A five-update run at tiny scale, under a minute on CPU. It exercises warm-start loading, the encoder, every advantage branch, and the evaluation suite:

```bash
python -c "import importlib.util,sys; s=importlib.util.spec_from_file_location('t','train-reinforce.py'); m=importlib.util.module_from_spec(s); sys.modules['t']=m; s.loader.exec_module(m); m.train_reinforce(ALGORITHM='MaxRL', TOTAL_UPDATES=5, EVAL_INTERVAL=5, LOG_INTERVAL=1, BATCH_SIZE=4, GROUP_SIZE=4, NUM_VAL_MAZES=10, TAG='smoke')"
```

`importlib` is needed because `train-reinforce.py` contains a hyphen and cannot be imported by name. Delete the resulting `checkpoints/maze_MaxRL_G4_8x8_s0_smoke.pth` afterward.

### 1. Train the BC warm-start

```bash
python train_behavior_cloning.py
```

Writes `checkpoints/maze_CNN.pth` plus a ladder of intermediate checkpoints `maze_CNN_s<step>.pth` every 50 gradient steps.

### 2. Promote a checkpoint to the warm-start

RL loads `checkpoints/BFS_BC_CNN-RL-starter.pth`. Promotion is deliberate rather than automatic, so that retraining BC never disturbs a sweep already in flight.

```bash
cp checkpoints/maze_CNN_s300.pth checkpoints/BFS_BC_CNN-RL-starter.pth
```

Pick from the ladder rather than taking the final epoch. The warm-start should be **deliberately mediocre** — clearly learning but far from saturated — so that RL has headroom to demonstrate anything. Epoch granularity is too coarse to hit that target: greedy success can jump from 18% to 52% across a single epoch boundary.

The one thing to rule out is a checkpoint that hasn't learned *when to stop*. Each expert trajectory contains exactly one `stop` out of roughly 9.5 actions, so a model that emits `P(stop) ≈ 0.105` on states far from the goal is still sitting at the dataset's class prior and has learned nothing about stopping. That compounds badly over a rollout. Prefer a checkpoint with `P(stop) ≲ 0.04` on start states.

A 3-channel checkpoint (`BFS_BC_CNN-RL-starter-3ch.pth`) is kept for the timestep-channel ablation. Runs with `IN_CHANNELS=3` load it automatically.

### 3. Run the RL sweep

```bash
python train-reinforce.py
```

This executes every entry in the `RUNS` list, one after another, each as its own W&B run. Failures are isolated per-run, so one bad config doesn't cost an overnight batch, and a manifest is rewritten after every run. Re-running skips configurations already recorded in the manifest, which makes the sweep safe to resume after an interruption.

### 4. Coverage curves (optional)

```bash
python coverage.py
```

Reads the sweep manifest, samples once per checkpoint, and writes per-algorithm pass@k figures to `assets/`. Raw per-maze success counts are saved alongside, so re-plotting at new `k` values costs nothing as long as `n` was large enough. Requires step 3 to have produced a manifest first.

---

## Outputs

| Artifact | Location |
|---|---|
| Trained weights | `checkpoints/maze_<algo>_G<G>_<D>x<D>_s<seed>[_<tag>].pth` |
| Sweep manifest | `checkpoints/runs_<SWEEP>.json` |
| Coverage figures and raw counts | `assets/` |
| Training curves | Weights & Biases, project `SURA` |

Run and checkpoint names encode everything that can differ between two runs in a sweep — algorithm, group size, learning rate, seed, critic flag, channel count. Anything left out would let one run silently overwrite another's weights.

Key metrics logged during training: `train_pass_1` and `train_pass_G` (pass@1 and pass@`G` on the training batch), `rolling_train_success_rate`, `policy_entropy`, `response_length`, `frac_K_zero` and `frac_K_all` (the fraction of groups where all rollouts failed or all succeeded), and the critic diagnostics.

---

## Running on Google Colab

RL training wants a GPU. The T4 tier is sufficient — the network is small and the loop is largely bound by host-side environment stepping, so a premium GPU buys much less than its cost.

```python
# 1. Confirm the GPU
!nvidia-smi

# 2. Mount Drive so checkpoints and the manifest survive a disconnect
from google.colab import drive
drive.mount('/content/drive')

# 3. Clone into Drive (first session only), then sync
import os
REPO = '/content/drive/MyDrive/SURA'
if not os.path.isdir(REPO):
    !git clone https://github.com/Hef50/SURA.git "{REPO}"
%cd /content/drive/MyDrive/SURA
!git fetch origin && git reset --hard origin/main

# 4. W&B
!pip -q install wandb
import os; os.environ["WANDB_DIR"] = "/content"   # keep wandb's many small files off Drive
import wandb; wandb.login()

# 5. Train
!python train-reinforce.py
```

Two things worth knowing. Git is unreliable on a Drive FUSE mount — it cannot preserve the stat metadata git caches, so `git pull` may silently fail and `git log` may report a broken branch even when the checkout is fine; `git fetch && git reset --hard origin/main` is the reliable sync, and verifying with `grep` on the actual file beats trusting git's output. And `WANDB_DIR` should point at local disk, because W&B writes thousands of small files and Drive is slow at that.

After a disconnect, re-run the mount, sync, and train cells. The manifest lives on Drive, so completed runs are skipped automatically.

---

## Results so far

**The timestep channel reduces greedy looping.** Comparing the 3-channel and 4-channel BC warm-starts at *matched* stochastic success (mean@1 of 37.9 vs 39.6 on 200 held-out mazes), the fraction of greedy rollouts that time out rather than solving fell from **0.705 to 0.535**. This is consistent with the mechanism: a stationary policy in a deterministic maze must cycle if it does not solve, and adding time to the state lets it break the cycle.

**RL comparisons are being re-run.** An earlier pass@k comparison across methods is not reported here because it was confounded — at the commit that produced those checkpoints, RLOO and GRPO computed advantages from a shaped reward while MaxRL used the binary success indicator, so the methods were not optimizing the same objective. Current runs use a single binary-reward path for all group methods.

**Earlier findings (behavior cloning, 8×8):** more data helps with strong diminishing returns past ~2k mazes; more parameters help with little gain past hidden dim 512; three MLP layers is the sweet spot; learning rate barely matters under Adam; and CNNs clearly outperform MLPs, which is why all subsequent work uses a convolutional policy.

---

## Background & References

- Williams (1992), *Simple statistical gradient-following algorithms for connectionist reinforcement learning* — REINFORCE.
- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.) — policy gradients, baselines, actor-critic (Ch. 13); multi-armed bandits (Ch. 2), since terminal-reward RL is essentially a contextual bandit.
- Shao et al. (2024), *DeepSeekMath* — GRPO; estimates the baseline from group scores instead of a critic.
- Ahmadian et al. (2024), *Back to Basics* — RLOO.
- Chen et al. (2021), *Evaluating Large Language Models Trained on Code* — the unbiased pass@k estimator.
- Tajwar, Arora, …, Zanette (2026), *MaxRL* — maximum-likelihood RL for verifiable-correctness tasks.

---

## Acknowledgments

This project is conducted as part of the **Summer Undergraduate Research Apprenticeship (SURA)** through Carnegie Mellon University's Office of Undergraduate Research and Scholar Development (OURSD), in **Prof. Andrea Zanette's lab**, under the day-to-day mentorship of PhD student **Daman Arora**. Thanks to the lab for guidance and for the MaxRL line of work that motivates this study.

---

*Maintained by Haresh Muralidharan · Carnegie Mellon University, ECE.*
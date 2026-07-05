# SURA Research Project: Investigating Value Functions in Terminal-Reward Reinforcement Learning

A research project investigating **whether learned value functions provide meaningful utility in terminal-reward (sparse-reward) reinforcement learning** — the regime that underlies modern RL for large language models (LLMs) — studied efficiently in gridworld maze environments.

> **Status:** Active research (Summer Undergraduate Research Apprenticeship, June–July 2026). Work in progress.
> Conducted in [Prof. Andrea Zanette's lab](https://azanette.com/) at Carnegie Mellon University, advised day-to-day by PhD student Daman Arora.

---

## Overview

Modern reinforcement learning for reasoning LLMs is dominated by **terminal-reward** methods: a model generates a full response and receives a single scalar reward at the end (e.g., 1 if the answer is correct, 0 otherwise). Increasingly, the field has moved toward **critic-free, group-based** optimization methods — GRPO, RLOO, and MaxRL — that estimate a baseline from a *group* of sampled rollouts rather than from a learned value function.

Value functions are a classical tool for reducing variance and improving credit assignment in RL, and they are well-established in dense-reward settings. Yet their utility in the terminal-reward regime remains **an open and actively debated question**: the field largely moved away from critics without a clean, controlled study of whether they help here.

This project builds a low-compute maze testbed to study that question directly, and to explore a specific hypothesis:

> **Group-based methods like MaxRL currently require multiple rollouts per prompt (a "group") to estimate their learning signal, which is computationally expensive. A learned value function could, in principle, provide that signal from a single rollout — potentially making MaxRL-style training substantially cheaper.**

---

## Research Questions

1. **Do learned value-function baselines/critics measurably improve performance** (sample efficiency, success rate, training stability) over critic-free policy-gradient methods in a controlled terminal-reward setting?
2. **How does any benefit depend on task difficulty, episode horizon, and reward sparsity?**
3. *(Exploratory)* **Can a value function substitute for group sampling in MaxRL-style training**, enabling single-rollout updates that retain MaxRL's advantages at lower cost?

---

## Approach

The project uses **procedurally generated maze environments** as a cheap, controlled proxy for terminal-reward RL. Mazes are small enough to run many controlled comparisons on a CPU or a free Colab GPU, while preserving the terminal-reward structure that characterizes LLM RL.

**Environment.** An agent navigates a randomly generated `D×D` maze (built with randomized Prim's algorithm) toward a goal, choosing from five actions `{up, down, left, right, stop}`. The reward is strictly terminal and binary: **+1 only if the agent issues `stop` while on the goal cell, and 0 otherwise** (including a premature `stop`, which ends the episode). Start and goal positions are randomized per maze; maze size is a tunable parameter.

**Methods implemented / under comparison:**
- **Behavior Cloning (BC)** — a supervised baseline that imitates a BFS shortest-path expert; also used to warm-start RL.
- **REINFORCE** — vanilla policy gradient.
- **REINFORCE with a value-function baseline** — the classical variance-reduction approach.
- **RLOO** (REINFORCE Leave-One-Out) — group baseline from the other samples.
- **GRPO** (Group Relative Policy Optimization) — group-normalized advantages.
- **MaxRL** — a maximum-likelihood, success-normalized group method.

**Evaluation.** Policies are compared on maze success rate (both greedy and stochastic pass@k-style rollouts), sample efficiency, training stability, gradient variance, policy entropy, mean reward, and response length. All experiments are tracked with [Weights & Biases](https://wandb.ai/).

---

## Repository Structure

```
.
├── maze_generation.py     # randomized Prim's maze generator + start/goal placement
├── environment.py         # the terminal-reward maze simulator (reset / step)
├── bfs_expert.py          # BFS shortest-path expert for behavior cloning
├── encodings.py           # maze-state -> vector encodings (compared experimentally)
├── model.py               # policy networks (MLP / CNN) and value head
├── train_bc.py            # behavior-cloning (supervised) training
├── algorithms/            # RL algorithms
│   ├── reinforce.py
│   ├── reinforce_baseline.py
│   ├── rloo.py
│   ├── grpo.py
│   └── maxrl.py
├── evaluate.py            # rollouts + metrics (greedy / stochastic success rates)
├── visualize.py           # renders mazes and policy behavior over training
└── README.md
```
*(Adjust to match your actual file layout.)*

---

## Setup

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install numpy torch matplotlib wandb
wandb login                     # authenticate for experiment tracking
```

For larger-scale runs, training can be moved to a Colab GPU runtime (clone the repo into Colab and set the runtime to GPU).

---

## Usage

```bash
# 1. Train a behavior-cloning baseline / warm-start checkpoint
python train_bc.py --grid_size 8 --epochs 1 --out model.pth

# 2. Train an RL method, warm-started from the BC checkpoint
python -m algorithms.reinforce --init model.pth --grid_size 8

# 3. Evaluate a trained policy
python evaluate.py --checkpoint <path> --num_mazes 500 --stochastic

# 4. Visualize policy behavior
python visualize.py --checkpoint <path>
```
*(Example commands — update flags to match your implementation.)*

---

## Background & References

Key methods and references informing this work:

- Williams (1992), *Simple statistical gradient-following algorithms for connectionist reinforcement learning* — REINFORCE.
- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.) — policy gradients, baselines, actor-critic (Ch. 13).
- Schulman et al. (2015), *High-Dimensional Continuous Control Using Generalized Advantage Estimation* — GAE.
- Shao et al. (2024), *DeepSeekMath* — GRPO (critic-free, group-normalized).
- Ahmadian et al. (2024), *Back to Basics: Revisiting REINFORCE-style Optimization for LLMs* — RLOO.
- Tajwar, Arora, …, Zanette (2026), *MaxRL* — maximum-likelihood RL for verifiable-correctness tasks.

*(Add exact links/citations as appropriate.)*

---

## Acknowledgments

This project is conducted as part of the **Summer Undergraduate Research Apprenticeship (SURA)** through Carnegie Mellon University's Office of Undergraduate Research and Scholar Development (OURSD), in **Prof. Andrea Zanette's lab**, under the day-to-day mentorship of PhD student **Daman Arora**. Thanks to the lab for guidance and for the MaxRL line of work that motivates this study.

---

*Maintained by Haresh Muralidharan · Carnegie Mellon University, ECE.*

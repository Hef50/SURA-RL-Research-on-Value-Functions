# Colab upload checklist

Upload **these files into one working directory** on Colab (Drive folder or `/content`), then run from that folder.

## Required for RL training (`train-reinforce.py`)

| File | Role |
|------|------|
| `train-reinforce.py` | main training loop |
| `environment.py` | `MazeEnv` + `VecMazeEnv` |
| `maze_encodings.py` | encodings (+ `encode_batch`) |
| `model.py` | `MazeCNN` |
| `evaluate.py` | validation metrics |
| `maze_generation.py` | maze gen + fixed val set |
| `BFS_BC_CNN-RL-starter.pth` | warm-start weights |

Locally the starter lives at `checkpoints/BFS_BC_CNN-RL-starter.pth`.  
On Colab you can either:

1. **Flat upload** — put the `.pth` next to the `.py` files (same folder), or  
2. **Keep structure** — upload a `checkpoints/` folder with the starter inside.

`train-reinforce.py` resolves both via `resolve_path()`.

## Optional (only if you retrain BC on Colab)

| File | Role |
|------|------|
| `train_behavior_cloning.py` | BC training |
| `bfs_expert.py` | BFS expert used by BC |

## Do **not** upload (legacy / artifacts)

Everything under `archive/` (old GRPO/MaxRL script copies, visualize helpers, `wandb-test.py`, `results.py`), extra trained weights you don't need, `assets/`, `.venv`, `wandb/`.

## Colab setup snippet

```python
# Runtime → Change runtime type → GPU (T4)
!pip install -q wandb

import os
os.chdir("/content/your_folder")  # or wherever you uploaded the files

# wandb.login()   # once per account, or:
# import os; os.environ["WANDB_MODE"] = "offline"

!python train-reinforce.py
```

Trained weights are written to `checkpoints/maze_<ALGORITHM>.pth` if that folder exists, otherwise to the current directory.

# In-Context Reinforcement Learning with Algorithm Distillation

This repository is an environment for running experiments with algorithm distillation. Link to work in this area: [arxiv](https://arxiv.org/abs/2210.14215).

---

## How to run the project?

| Tool | Version |
| ------------- | ----------------------------------- |
| Python | Python 3.10 |
| CUDA Toolkit | 12.8 |
| nvcc Compiler | 12.8.61 |
| CUDA Build | cuda_12.8.r12.8/compiler.35404655_0 |

Installing the CUDA toolchain is optional, and I'll leave that for you to explore.

###### 1. Create a virtual environment

If you only have one Python interpreter installed, you can create a virtual environment with the command:
```bash
# In-Context Reinforcement Learning with Algorithm Distillation

A sandbox for experiments on algorithm distillation in reinforcement learning. The codebase provides small environments, experiment configs, and training/evaluation scripts to study how learning algorithms can be distilled into model behavior.

Features
- Experiment runner: pick an experiment config and execute its `experiment.py` via `main.py`.
- Multiple example environments under `experiments/` (DarkRoom, KArmedBandit, etc.).
- Simple tracking integration using MLflow (optional).

Requirements
- Python 3.10 or newer
- See `pyproject.toml` for the primary runtime dependencies (torch, mlflow, numpy, pyyaml, questionary, rich, tqdm).

Quickstart
1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies. If you have a `requirements.txt`, use it; otherwise install from `pyproject.toml`'s dependencies:

```bash
# If you have requirements.txt
pip install -r requirements.txt

# Or install the main runtime deps directly
pip install dotenv mlflow numpy pyyaml questionary rich torch tqdm
```

3. Run the interactive experiment chooser:

```bash
python main.py
```

You can pass CLI flags to `main.py`:

- `--experiment` / `-e` : experiment name (matches `experiment_name` in an `experiments/*/config.yaml`)
- `--seed` / `-s` : random seed
- `--autoconfirm` / `-a` : skip interactive confirmation
- `--track` / `-t` : enable experiment tracking (MLflow)
- `--run_name` / `-r` : override run name when tracking

Example (run an experiment non-interactively):

```bash
python main.py -e DarkRoomEasy -s 0 -a -t -r "test-run"
```

MLflow (optional)
- To enable experiment tracking, start an MLflow server and point runs at it. Example local server:

```bash
mkdir -p mlflow_data/data_local mlflow_data/artifacts
mlflow server \
	--backend-store-uri "file:///$PWD/mlflow_data/data_local" \
	--default-artifact-root "file:///$PWD/mlflow_data/artifacts" \
	--host localhost --port 5000
```

Project layout (top-level)
- `main.py` — experiment chooser and runner
- `experiments/` — per-experiment directories with `config.yaml` and `experiment.py`
- `src/` — core library (environments, agents, datasets, transformers, workflow utils)
- `tests/` — unit tests

Running tests

```bash
pip install pytest
pytest -q
```

Notes
- CUDA is optional. If you have GPUs available and `torch` built with CUDA, the code will use `cuda` automatically.
- The repository is a research sandbox; configs and scripts may assume specific hyperparameters and are intended for experimentation rather than production use.

If you want, I can also update `pyproject.toml` or add a `requirements.txt` and a short CONTRIBUTING guideline. Tell me which you'd prefer.
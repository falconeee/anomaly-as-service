# Phase 0: Project foundation

This phase prepares the development environment: an installable Python package, dependencies with
locked versions and tools that check the code automatically. Everything that comes after
(training, API, Docker, CI) depends on this base to be reproducible on any machine.

## Main files

| File | Role |
|---|---|
| `pyproject.toml` | Package name, dependencies and tool configuration |
| `uv.lock` | Exact version and hash of every installed package (generated automatically) |
| `.python-version` | Python version used (3.12) |
| `src/anomaly_service/` | Package code (the model is in `model/mscvae.py`) |
| `.pre-commit-config.yaml` | Checks run before each commit |
| `Makefile` | Shortcuts for the most common commands (`make help` lists them all) |

## Managing the environment with uv

[uv](https://docs.astral.sh/uv/) creates the virtual environment, installs the dependencies and runs
commands inside it. The commands used in the project:

| Command | What it does |
|---|---|
| `uv venv --python 3.12` | Creates the `.venv` virtual environment (downloads Python if needed) |
| `uv add package` | Adds a dependency to `pyproject.toml`, updates `uv.lock` and installs it |
| `uv add --dev package` | The same, in the development tools group |
| `uv sync` | Makes `.venv` identical to `uv.lock` (installs what is missing, removes what is extra) |
| `uv sync --frozen` | Installs exactly `uv.lock`, without recomputing anything (the mode used in CI and Docker) |
| `uv run command` | Runs a command inside the environment |

### Why `uv.lock` exists

`pyproject.toml` declares version **ranges** (for example, `torch>=2.14.1`). `uv.lock`
records the **exact version and hash** of every package, including indirect dependencies. With
it, the development machine, CI and the Docker image install exactly the same files.
A single `uv.lock` serves both macOS and Linux.

### PyTorch without CUDA on Linux

On Linux, the PyTorch published on PyPI includes NVIDIA GPU libraries that add up to several GB.
The project does not use a GPU in production, so `pyproject.toml` points Linux PyTorch to
PyTorch's own *CPU-only* package index. On macOS, it keeps the default package, which uses the Mac's
GPU (MPS). The index is declared with `explicit = true`, meaning it is queried **only** for
PyTorch; this prevents a third-party index from supplying other dependencies by mistake.

In `uv.lock`, this shows up as `torch 2.14.1+cpu` for Linux and `torch 2.14.1` for macOS.

### Dependency groups

| Group | Content | Where it is installed |
|---|---|---|
| main | Everything the API and the worker need (torch, pandas, FastAPI, Celery, mlflow-skinny...) | Always |
| `dev` | pytest, ruff, mypy, pre-commit, fakeredis, full mlflow (local web UI) | Only on the development machine |
| `train` | matplotlib (training plots) | Development and the trainer image |

`uv sync` installs all three groups. The Docker images install only what they need.

## Code organization (src layout)

The code lives in `src/anomaly_service/`, not at the repository root. This way, the package can only be
imported after it is installed (`uv sync` does that). The tests then use the installed package,
as happens in production, instead of working by accident because they are in the right folder.

## Automated checks

| Tool | Checks |
|---|---|
| ruff | Common errors, imports, style and formatting |
| mypy | Types (for example, passing text where a number is expected) |
| pytest | Automated tests |
| pre-commit | Runs ruff, formatting and `uv lock --check` before each commit |

The pre-commit hooks call the tools via `uv run`, using the same versions as `uv.lock`.
The `uv lock --check` hook prevents committing a changed `pyproject.toml` without updating the lock.

## Design choices

- uv as the package manager: a single `uv.lock` serves macOS and Linux, allows choosing the
  PyTorch index per operating system and installs in seconds. Alternatives such as
  `pip` + `requirements.txt` or Poetry would require separate files per platform or more
  configuration.
- The logging module is called `logger.py` (not `logging.py`) so that it does not share its name with the
  standard library's `logging` module.
- The model's original file (`src/MSCVAE.py`) was moved to
  `src/anomaly_service/model/mscvae.py`, and the broken import `from src.utils import logger` was
  replaced by the new logging module.

## Step-by-step reproduction

Prerequisites: [uv](https://docs.astral.sh/uv/getting-started/installation/) (on macOS:
`brew install uv`) and `make`.

1. Create the environment and install the dependencies:
   ```bash
   make setup
   ```
   Expected result: `.venv` folder created and pre-commit hooks installed.
2. Run every check:
   ```bash
   make check
   ```
   Expected result: `All checks passed!` (ruff), `Success: no issues found` (mypy) and every
   test passing.
3. (Optional) Check the PyTorch installation:
   ```bash
   uv run python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
   ```
   On macOS with Apple Silicon, the second value is `True`.

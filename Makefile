.DEFAULT_GOAL := help
.PHONY: help setup lint format typecheck test check \
        mlflow train sample-csv train-smd status serve request examples \
        up down logs train-docker e2e monitoring clean-docker

# Local (non-Docker) runs track experiments in a SQLite file at the repo root.
MLFLOW_TRACKING_URI ?= sqlite:///mlflow.db
export MLFLOW_TRACKING_URI
export MLFLOW_DISABLE_AGENT_HINT = 1
GIT_SHA ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)
export GIT_SHA

help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# --- Development -------------------------------------------------------------------------
setup: ## Create the venv, install locked deps and git hooks
	uv venv --python 3.12 --allow-existing
	uv sync --frozen
	uv run pre-commit install

lint: ## Lint and check formatting
	uv run ruff check .
	uv run ruff format --check .

format: ## Auto-fix lint issues and format code
	uv run ruff check --fix .
	uv run ruff format .

typecheck: ## Static type checking
	uv run mypy src

test: ## Run the test suite with coverage
	uv run pytest --cov=anomaly_service --cov-report=term-missing

check: lint typecheck test ## Everything CI runs (except the Docker e2e)

# --- Local MLOps loop (no Docker) --------------------------------------------------------
mlflow: ## MLflow UI on http://localhost:5050 (local SQLite store; 5000 is AirPlay on macOS)
	uv run mlflow ui --backend-store-uri $(MLFLOW_TRACKING_URI) --port 5050

train: ## Train, register and run the promotion gate (CONFIG=configs/train_synthetic.yaml)
	uv run python -m anomaly_service.training.train --config $(or $(CONFIG),configs/train_synthetic.yaml) --register --promote

sample-csv: ## Write datasets/exemplo.csv, a sample in the "own data" CSV layout
	uv run python -m anomaly_service.data.csv_source datasets/exemplo.csv

train-smd: ## Train on SMD machine-1-1 (downloads ~30 MB), register and promote
	uv run python -m anomaly_service.training.train --config configs/train_smd.yaml --register --promote

status: ## Registered versions, aliases and metrics (MODEL=mscvae-synthetic)
	uv run python -m anomaly_service.training.promote status --model $(or $(MODEL),mscvae-synthetic)

serve: ## API on http://localhost:8000 serving @champion (PORT=8001 if the Docker stack is up)
	uv run uvicorn anomaly_service.api.main:create_app --factory --reload --port $(or $(PORT),8000)

request: ## Build a request body from a config's test split (CONFIG=..., OUT=request.json)
	uv run python -m anomaly_service.data.examples --config $(CONFIG) --out $(or $(OUT),request.json)

examples: ## Regenerate example request bodies (synthetic + SMD)
	uv run python -m anomaly_service.data.examples --config configs/train_synthetic.yaml --out examples/request_synthetic.json
	uv run python -m anomaly_service.data.examples --config configs/train_smd.yaml --start 15500 --rows 1500 --out examples/request_smd.json

# --- Docker ------------------------------------------------------------------------------
up: ## Build and start the stack (API, worker, MLflow, Postgres, Redis)
	docker compose up -d --build --wait

down: ## Stop the stack (keeps volumes)
	docker compose --profile train --profile monitoring down

logs: ## Follow API and worker logs
	docker compose logs -f api worker

train-docker: ## Train/register/promote inside Docker (CONFIG=configs/train_synthetic.yaml)
	docker compose --profile train run --rm --build trainer train --config $(or $(CONFIG),configs/train_synthetic.yaml) --register --promote

e2e: ## End-to-end smoke test against the running stack
	./scripts/e2e.sh

monitoring: ## Start Prometheus (:9090) and Grafana (:3000)
	docker compose --profile monitoring up -d

clean-docker: ## Stop everything and DELETE volumes (registry, artifacts, queue)
	docker compose --profile train --profile monitoring down -v

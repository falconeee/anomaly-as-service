.DEFAULT_GOAL := help
.PHONY: help setup lint format typecheck test check

help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

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

check: lint typecheck test ## Everything CI runs

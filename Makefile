.PHONY: help install test test-coverage lint format clean build docs


# Environment setup
install: ## Install dependencies with uv
	uv sync --all-extras
	@echo "Dependencies installed"

install-dev: ## Install development dependencies
	uv sync --all-extras --dev
	uv pip install -e .
	@echo "Development environment ready"

check-deps: ## Check for missing dependencies
	uv pip check
	@echo "All dependencies satisfied"


# Code quality
lint: ## Run linting checks
	uv run ruff check src/ tests/

lint-fix: ## Fix linting issues automatically
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/

# Security scanning
security: ## Run bandit security scan
	uv run bandit -r src/ -ll

# Testing commands
test: ## Run all tests (unit + integration + e2e)
	@echo "Running comprehensive test suite..."
	uv run pytest tests/ -v
.PHONY: help install test test-coverage lint format clean build docs


# Environment setup
install: ## Install dependencies with uv
	uv sync
	@echo "Dependencies installed"

install-dev: ## Install development dependencies
	uv sync --extra dev
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

test-coverage: ## Run tests with coverage report
	@echo "Running tests with coverage..."
	uv run pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html --cov-report=xml
	@echo "Coverage report generated in htmlcov/ directory"

# Code formatting
format: ## Format code with ruff
	uv run ruff format src/ tests/
	@echo "Code formatted"

# Build commands
build: ## Build the package
	@echo "Building package..."
	uv build
	@echo "Package built in dist/ directory"

# Clean commands
clean: ## Clean build artifacts and cache
	@echo "Cleaning build artifacts..."
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Clean completed"

# Documentation
docs: ## Generate documentation (if available)
	@echo "Generating documentation..."
	@if [ -f "docs/conf.py" ]; then \
		cd docs && make html; \
	else \
		echo "No documentation setup found. Consider adding Sphinx documentation."; \
	fi

# Help command
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
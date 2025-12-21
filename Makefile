# LexiAI Makefile - Test & Development Commands

.PHONY: help test test-unit test-integration test-cov test-fast test-security test-perf clean-test install-test setup-test dev

# Default target
help:
	@echo "LexiAI Test & Development Commands"
	@echo "===================================="
	@echo ""
	@echo "Testing:"
	@echo "  make test              - Run all tests with coverage"
	@echo "  make test-unit         - Run unit tests only"
	@echo "  make test-integration  - Run integration tests only"
	@echo "  make test-fast         - Run tests without coverage (faster)"
	@echo "  make test-cov          - Run tests and open coverage report"
	@echo "  make test-security     - Run security tests only"
	@echo "  make test-perf         - Run performance tests only"
	@echo "  make test-watch        - Run tests in watch mode"
	@echo ""
	@echo "Integration Testing:"
	@echo "  make test-bash         - Run bash integration test"
	@echo "  make test-e2e          - Run end-to-end tests"
	@echo ""
	@echo "Setup:"
	@echo "  make install-test      - Install test dependencies"
	@echo "  make setup-test        - Setup test environment"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean-test        - Clean test artifacts"
	@echo "  make clean-cov         - Clean coverage reports"
	@echo ""
	@echo "Development:"
	@echo "  make lint              - Run code linting"
	@echo "  make format            - Format code with black"
	@echo "  make typecheck         - Run type checking with mypy"
	@echo "  make dev               - Start API server (api-only) with venv + .env"
	@echo ""

# ============================================================================
# Test Commands
# ============================================================================

# Run all tests with coverage
test:
	@echo "Running all tests with coverage..."
	pytest tests/ -v --cov=backend --cov-report=html --cov-report=term-missing

# Run unit tests only
test-unit:
	@echo "Running unit tests..."
	pytest tests/ -v -m "not integration" --cov=backend

# Run integration tests only
test-integration:
	@echo "Running integration tests..."
	pytest tests/integration/ -v

# Run tests without coverage (faster)
test-fast:
	@echo "Running tests (fast mode)..."
	pytest tests/ -v --no-cov

# Run tests and open coverage report
test-cov: test
	@echo "Opening coverage report..."
	@open htmlcov/index.html || xdg-open htmlcov/index.html || start htmlcov/index.html

# Run security tests only
test-security:
	@echo "Running security tests..."
	pytest tests/ -v -m security
	@echo "Running bandit security scan..."
	bandit -r backend/ -ll

# Run performance tests only
test-perf:
	@echo "Running performance tests..."
	pytest tests/ -v -m performance --benchmark-only

# Run tests in watch mode (requires pytest-watch)
test-watch:
	@echo "Running tests in watch mode..."
	ptw tests/ -- -v --no-cov

# ============================================================================
# Integration Testing
# ============================================================================

# Run bash integration test
test-bash:
	@echo "Running bash integration test..."
	@echo "Make sure API server is running (python start_middleware.py)"
	@./scripts/test_auth_profile_integration.sh

# Run end-to-end tests
test-e2e: test-integration test-bash
	@echo "End-to-end tests completed!"

# ============================================================================
# Setup & Installation
# ============================================================================

# Install test dependencies
install-test:
	@echo "Installing test dependencies..."
	pip install -r tests/requirements-test.txt

# Setup test environment
setup-test: install-test
	@echo "Setting up test environment..."
	@mkdir -p tests/logs
	@mkdir -p htmlcov
	@echo "Test environment ready!"

# ============================================================================
# Cleanup
# ============================================================================

# Clean test artifacts
clean-test:
	@echo "Cleaning test artifacts..."
	@rm -rf .pytest_cache
	@rm -rf tests/.pytest_cache
	@rm -rf tests/__pycache__
	@rm -rf tests/integration/__pycache__
	@rm -rf tests/logs/*.log
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@echo "Test artifacts cleaned!"

# Clean coverage reports
clean-cov:
	@echo "Cleaning coverage reports..."
	@rm -rf htmlcov
	@rm -rf .coverage
	@rm -rf coverage.xml
	@echo "Coverage reports cleaned!"

# Clean all
clean: clean-test clean-cov
	@echo "All clean!"

# ============================================================================
# Development
# ============================================================================

# Start API server with venv + .env (api-only)
dev:
	@echo "Starting LexiAI API server (api-only)..."
	@bash -lc 'if [ -d ".venv" ]; then source .venv/bin/activate; \
	elif [ -d "venv" ]; then source venv/bin/activate; \
	else echo "No virtual environment found (.venv or venv)."; exit 1; fi; \
	if [ -f ".env" ]; then export $$(grep -v "^#" .env | xargs); fi; \
	python3 start_middleware.py --api-only --no-browser'

# Run linting
lint:
	@echo "Running linters..."
	flake8 backend/ tests/ --max-line-length=120
	pylint backend/ --max-line-length=120

# Format code
format:
	@echo "Formatting code..."
	black backend/ tests/ --line-length=120
	isort backend/ tests/

# Type checking
typecheck:
	@echo "Running type checker..."
	mypy backend/ --ignore-missing-imports

# Run all checks
check: format lint typecheck test
	@echo "All checks passed!"

# ============================================================================
# CI/CD Simulation
# ============================================================================

# Simulate CI pipeline
ci: clean-test install-test test lint typecheck
	@echo "CI pipeline completed!"

# ============================================================================
# Coverage Reporting
# ============================================================================

# Generate coverage badge
coverage-badge:
	@echo "Generating coverage badge..."
	coverage-badge -o coverage.svg -f

# Upload coverage to codecov (requires CODECOV_TOKEN)
coverage-upload:
	@echo "Uploading coverage to Codecov..."
	codecov

# ============================================================================
# Test Data Management
# ============================================================================

# Generate test fixtures
generate-fixtures:
	@echo "Generating test fixtures..."
	python tests/generate_fixtures.py

# ============================================================================
# Performance Benchmarking
# ============================================================================

# Run benchmark suite
benchmark:
	@echo "Running benchmark suite..."
	pytest tests/ --benchmark-only --benchmark-autosave

# Compare benchmarks
benchmark-compare:
	@echo "Comparing benchmarks..."
	pytest-benchmark compare

# ============================================================================
# Documentation
# ============================================================================

# Generate test documentation
docs-test:
	@echo "Generating test documentation..."
	pytest tests/ --html=tests/reports/test-report.html --self-contained-html

# ============================================================================
# Docker Support
# ============================================================================

# Run tests in Docker
test-docker:
	@echo "Running tests in Docker..."
	docker-compose -f docker-compose.test.yml up --abort-on-container-exit

# ============================================================================
# Parallel Testing
# ============================================================================

# Run tests in parallel (requires pytest-xdist)
test-parallel:
	@echo "Running tests in parallel..."
	pytest tests/ -n auto -v

# ============================================================================
# Specific Test Files
# ============================================================================

# Test authentication
test-auth:
	@echo "Testing authentication..."
	pytest tests/test_authentication.py -v

# Test profile builder
test-profile-builder:
	@echo "Testing profile builder..."
	pytest tests/test_profile_builder.py -v

# Test profile context
test-profile-context:
	@echo "Testing profile context..."
	pytest tests/test_profile_context.py -v

# Test full flow
test-flow:
	@echo "Testing full auth + profile flow..."
	pytest tests/integration/test_auth_profile_flow.py -v

# ============================================================================
# Debugging
# ============================================================================

# Run tests with debugger
test-debug:
	@echo "Running tests with debugger..."
	pytest tests/ -v --pdb

# Run specific test with verbose output
test-one:
	@echo "Run with: make test-one TEST=tests/test_file.py::TestClass::test_method"
	pytest $(TEST) -vv -s

# ============================================================================
# Hooks Integration
# ============================================================================

# Run pre-test hooks
hooks-pre:
	@echo "Running pre-test hooks..."
	npx claude-flow@alpha hooks pre-task --description "Running LexiAI test suite"

# Run post-test hooks
hooks-post:
	@echo "Running post-test hooks..."
	npx claude-flow@alpha hooks post-task --task-id "lexiai-tests"

# Test with hooks
test-with-hooks: hooks-pre test hooks-post
	@echo "Tests completed with hooks!"

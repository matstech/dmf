.PHONY: help lock install build test test-integration benchmark-ltm-local chroma-up chroma-down check compile docs-serve docs-build

# Update poetry.lock
lock:
	poetry lock

# Install dependencies (including dev) and download the spaCy model
install:
	poetry install --with dev --no-interaction
	poetry run python -m spacy download en_core_web_sm

# Verify package metadata
check:
	poetry check

# Compile Python source files (syntax check)
compile:
	poetry run python -m compileall dmf tests integrationtest

# Run the test suite
test:
	poetry run pytest

# Start Chroma 0.6.3 and wait for the service to be ready
chroma-up:
	docker compose -f compose.chroma.yml up -d --wait

# Run integration tests requiring the local Chroma server
test-integration:
	poetry run pytest -o addopts="-v --tb=short" -m integration integrationtest

# Run the local DMF LTM + Ollama mini-benchmark (independent of pytest)
benchmark-ltm-local:
	poetry run python -m integrationtest.run_ltm_benchmark

# Stop containers without removing persistent volumes
chroma-down:
	docker compose -f compose.chroma.yml down

# Build the wheel distribution package in dist/
build:
	poetry build --format wheel

# Start the documentation development server
docs-serve:
	poetry run mkdocs serve

# Build static documentation
docs-build:
	poetry run mkdocs build

# Show available targets and their descriptions
help:
	@echo "Available targets in Makefile:"
	@echo ""
	@echo "  make lock                  Update poetry.lock"
	@echo "  make install               Install dependencies (including dev) and download spaCy model"
	@echo "  make check                 Verify package metadata"
	@echo "  make compile               Compile Python source files"
	@echo "  make test                  Run the unit test suite"
	@echo "  make chroma-up             Start Chroma 0.6.3 (Docker container)"
	@echo "  make test-integration      Run integration tests against the local Chroma server"
	@echo "  make benchmark-ltm-local   Run the local DMF LTM + Ollama benchmark"
	@echo "  make chroma-down           Stop Chroma containers"
	@echo "  make build                 Build wheel package in dist/"
	@echo "  make docs-serve            Start development server for documentation"
	@echo "  make docs-build            Build static documentation"

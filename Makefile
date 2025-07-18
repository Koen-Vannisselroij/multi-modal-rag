# Makefile for Multimodal Agentic RAG

VENV := .venv
UV := $(VENV)/bin/uv

.PHONY: help init run test clean test-retrieval

help:  ## Show this help message and exit
	@awk 'BEGIN {FS = ":.*?## "}; /^[a-zA-Z0-9_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

init:  ## Set up the virtual environment and install dependencies
	uv venv .venv
	uv sync
	@echo "Entering virtual environment. Type 'exit' to leave."
	@. .venv/bin/activate; bash

run:  ## Run the main application
	python src/main.py

test:  ## Run all tests with pytest
	pytest tests/

EVAL_CASES ?= data/evaluate/cases/evaluate_retrieval_base.yaml

evaluate-retrieval:  ## Run the retrieval accuracy evaluation script
	@if [ ! -f $(EVAL_CASES) ]; then \
		echo "Error: $(EVAL_CASES) not found. Please provide a test cases YAML file."; \
		exit 1; \
	fi
	@echo "Running retrieval accuracy evaluation with test cases: $(EVAL_CASES)"
	@echo "Test cases file: $(EVAL_CASES)"
	@echo "Data folder: data/evaluate/data/"
	@echo "Cases folder: data/evaluate/cases/"
	PYTHONPATH=src python scripts/evaluate_retrieval_accuracy.py $(EVAL_CASES)

clean:  ## Remove caches and the virtual environment
	rm -rf __pycache__ .pytest_cache .venv
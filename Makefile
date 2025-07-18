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

CONFIG ?= example_config.yaml

run:  ## Run the main application (optionally specify CONFIG=your_config.yaml)
	@if [ ! -f $(CONFIG) ]; then \
		echo "Error: $(CONFIG) not found. Please provide a config YAML file."; \
		exit 1; \
	fi
	python src/main.py $(CONFIG)

test:  ## Run all tests with pytest
	pytest tests/

EVAL_CASES ?= data/evaluate/cases/evaluate_text_retrieval_base.yaml

CONFIG_EVAL ?= example_config.yaml

evaluate-text-retrieval:  ## Run the text retrieval accuracy evaluation script
	@if [ ! -f $(EVAL_CASES) ]; then \
		echo "Error: $(EVAL_CASES) not found. Please provide a test cases YAML file."; \
		exit 1; \
	fi
	@if [ ! -f $(CONFIG_EVAL) ]; then \
		echo "Error: $(CONFIG_EVAL) not found. Please provide a config YAML file."; \
		exit 1; \
	fi
	@echo "Running text retrieval accuracy evaluation with test cases: $(EVAL_CASES) and config: $(CONFIG_EVAL)"
	PYTHONPATH=src python scripts/evaluate_text_retrieval_accuracy.py $(EVAL_CASES) $(CONFIG_EVAL)

clean:  ## Remove caches and the virtual environment
	rm -rf __pycache__ .pytest_cache .venv
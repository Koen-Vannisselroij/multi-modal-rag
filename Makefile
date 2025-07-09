# Makefile for Multimodal Agentic RAG

VENV := .venv
UV := $(VENV)/bin/uv

.PHONY: setup run test clean lock

init:
	uv venv $(VENV)
	uv sync
	@echo "Entering virtual environment. Type 'exit' to leave."
	@. $(VENV)/bin/activate; bash

run:
	python src/main.py

test:
	pytest tests/

clean:
	rm -rf __pycache__ .pytest_cache .venv
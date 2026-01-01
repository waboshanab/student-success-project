# Makefile for common tasks
.PHONY: help install test lint format clean

help:
	@echo "Available targets: install, test, lint, format, clean"

install:
	pip install -r requirements.txt

test:
	pytest -q

lint:
	flake8 .

format:
	black .

clean:
	rm -rf .pytest_cache __pycache__ build dist

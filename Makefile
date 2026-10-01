.PHONY: install install-dev test lint download-data train

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/

lint:
	ruff check src/ tests/ scripts/
	black --check src/ tests/ scripts/

download-data:
	python scripts/download_data.py

train:
	python scripts/train.py --config configs/default.yaml

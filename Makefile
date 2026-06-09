.PHONY: install train test lint format typecheck run demo evaluate score docker all check

install:
	pip install -e ".[dev,explain]"

train:
	python scripts/train_models.py

evaluate:
	python scripts/evaluate.py

test:
	pytest --cov --cov-report=term-missing

lint:
	ruff check src tests scripts

format:
	ruff format src tests scripts

typecheck:
	mypy src

run:
	uvicorn product_intelligence.api.main:app --reload

demo:
	python scripts/demo_requests.py

score:
	python -m product_intelligence.pipelines.daily_batch_scoring

docker:
	docker build -t pulse360-product-intelligence:latest .

check: lint format typecheck test

all: install train check

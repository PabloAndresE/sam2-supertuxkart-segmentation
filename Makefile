.PHONY: install dev test lint train eval eval-test ablations seeds bench app docker-app clean

install:
	pip install -e .

dev:
	pip install -e ".[app,dev]"

test:
	pytest -q

lint:
	ruff check src tests scripts

train:
	python -m stkseg.train --config configs/default.yaml

eval:
	python -m stkseg.evaluate --config configs/default.yaml --split val

# Deliberately separate: the test split is a one-shot measurement.
eval-test:
	python -m stkseg.evaluate --config configs/default.yaml --split test --confirm-final

ablations:
	python scripts/run_experiments.py --suite configs/ablations.yaml

seeds:
	python scripts/run_experiments.py --suite configs/seeds.yaml

bench:
	python scripts/benchmark.py --out docs/assets/benchmark.json

app:
	python app/app.py

docker-app:
	docker compose up --build

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__

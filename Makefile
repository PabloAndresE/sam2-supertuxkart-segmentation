.PHONY: install dev test lint train eval eval-test ablations seeds bench demo app docker-app clean

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

demo:
	python -m stkseg.live --frames seg_dataset --track abyss --start 100 --count 60 \
	  --size 384 --scale 0.7 --gif docs/assets/demo/abyss_live.gif \
	  --mp4 docs/assets/demo/abyss_live.mp4

app:
	python app/app.py

docker-app:
	docker compose up --build

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__

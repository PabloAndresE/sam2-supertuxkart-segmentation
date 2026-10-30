.PHONY: install dev test lint train eval app docker-app clean

install:
	pip install -e .

dev:
	pip install -e ".[app,dev]"

test:
	pytest -q

lint:
	ruff check src tests

train:
	python -m stkseg.train --config configs/default.yaml

eval:
	python -m stkseg.evaluate --config configs/default.yaml

app:
	python app/app.py

docker-app:
	docker compose up --build

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__

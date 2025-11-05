.PHONY: help env install test lint format clean docker-build docker-run docs
.PHONY: api compose-api cli-help

help:
	@echo "Available targets:"
	@echo "  make env          - Create conda environment"
	@echo "  make install      - Install package in editable mode"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linters"
	@echo "  make format       - Format code"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run Docker container"
	@echo "  make docs         - Build documentation"
	@echo "  make api          - Run FastAPI server locally (uvicorn)"
	@echo "  make compose-api  - Run API via docker-compose"
	@echo "  make cli-help     - Show CLI commands"

env:
	conda env create -f environment.yml
	conda activate pde_solver && pip install -e .

install:
	pip install -e .

test:
	pytest tests/ -v

lint:
	ruff check pde_solver/ tests/
	mypy pde_solver/ --ignore-missing-imports
	black --check pde_solver/ tests/
	isort --check pde_solver/ tests/

format:
	black pde_solver/ tests/
	isort pde_solver/ tests/
	ruff check --fix pde_solver/ tests/

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

docker-build:
	docker build -t pde-solver:latest -f Dockerfile .

docker-run:
	docker run --rm -it -v $(PWD):/workspace pde-solver:latest

docs:
	cd docs && make html

api:
	uvicorn pde_solver.api.server:app --host 0.0.0.0 --port 8000 --reload

compose-api:
	docker compose up api

cli-help:
	python run_solver.py --help



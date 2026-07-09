PYTHON ?= python

.PHONY: test lint typecheck run-toy clean

test:
	pytest

lint:
	ruff check src tests

typecheck:
	mypy src tests

run-toy:
	$(PYTHON) -m ttt_cache_lab.cli run --config configs/feasibility_toy.yaml

summarize-toy:
	$(PYTHON) -m ttt_cache_lab.cli summarize --input runs/feasibility-toy/summary.csv

first-table-toy:
	$(PYTHON) -m ttt_cache_lab.cli first-table --input runs/feasibility-toy/summary.csv

sweep-toy:
	$(PYTHON) -m ttt_cache_lab.cli sweep --config configs/sweep_toy_update_norm.yaml

run-hf-tiny:
	$(PYTHON) -m ttt_cache_lab.cli run --config configs/feasibility_hf_tiny.yaml

clean:
	rm -rf runs outputs .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info src/*.egg-info

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

clean:
	rm -rf runs outputs .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info src/*.egg-info

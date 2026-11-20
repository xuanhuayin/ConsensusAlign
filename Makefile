PYTHON ?= python

.PHONY: install test lint smoke build dry-run
install:
	$(PYTHON) -m pip install -e ".[data,dev]"
test:
	$(PYTHON) -m pytest
lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
smoke:
	$(PYTHON) examples/smoke.py --output smoke_output
build:
	$(PYTHON) -m build
dry-run:
	consensusalign-reproduce --plan configs/reproduce_coco.json --dry-run

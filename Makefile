.PHONY: setup test audit reproduce bundle-inventory package

PYTHON ?= python

setup:
	$(PYTHON) -m pip install -e '.[dev]'

test:
	$(PYTHON) -m pytest

audit:
	$(PYTHON) scripts/validate_inventory.py
	$(PYTHON) -m convex_solutions audit

reproduce:
	$(PYTHON) -m convex_solutions run all --output artifacts --seed 0

bundle-inventory:
	$(PYTHON) scripts/package_inventory.py

package: bundle-inventory
	$(PYTHON) scripts/package_repository.py

.PHONY: setup test audit reproduce results check-results bundle-inventory package

PYTHON ?= python

setup:
	$(PYTHON) -m pip install -e '.[dev]'

test:
	$(PYTHON) -m pytest

audit:
	$(PYTHON) scripts/validate_inventory.py
	$(PYTHON) -m convex_solutions audit

reproduce:
	$(PYTHON) scripts/reproduce_results.py

results:
	$(PYTHON) scripts/render_results.py

check-results:
	$(PYTHON) scripts/render_results.py --check

bundle-inventory:
	$(PYTHON) scripts/package_inventory.py

package: bundle-inventory
	$(PYTHON) scripts/package_repository.py

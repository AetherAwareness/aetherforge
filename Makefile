.PHONY: help install doctor test dryrun flagship flagship-live broad status quickstart init data validate lint clean dashboard demo recipes

PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip

help:
	@echo "AetherForge targets:"
	@echo "  make install      - venv + editable install with dev deps"
	@echo "  make doctor       - dependency check (use: aetherforge doctor --human)"
	@echo "  make status       - dashboard / Vast / Flash / next steps"
	@echo "  make quickstart   - doctor + dry-run in one shot"
	@echo "  make recipes      - list named --recipe presets"
	@echo "  make test         - unit + integration tests"
	@echo "  make dryrun       - smoke dry-run (recipe: dryrun)"
	@echo "  make broad        - Flash BROAD multi-sector dry-run"
	@echo "  make flagship     - logistics A3B flagship dry-run"
	@echo "  make flagship-live - flagship live train (needs GPU)"
	@echo "  make dashboard    - Training Console"
	@echo "  make data         - DataForge for example logistics domain"
	@echo "  make validate     - config schema check"
	@echo "  make demo         - capture README/HF screenshots + gif/webm"
	@echo "  make clean        - remove artifacts and caches"

install:
	python3 -m venv .venv
	$(PIP) install -U pip
	$(PIP) install -e ".[dev]"

doctor:
	$(PYTHON) -m aetherforge.cli doctor --human

status:
	$(PYTHON) -m aetherforge.cli status

recipes:
	$(PYTHON) -m aetherforge.cli recipes

quickstart:
	$(PYTHON) -m aetherforge.cli quickstart

test:
	$(PYTHON) -m pytest tests/ -q --tb=short

dryrun:
	$(PYTHON) -m aetherforge.cli train --recipe dryrun --dry-run --quiet

broad:
	$(PYTHON) -m aetherforge.cli train --recipe broad-flash --dry-run --quiet

flagship:
	bash scripts/run_flagship.sh dry-run

flagship-live:
	bash scripts/run_flagship.sh live

dashboard:
	$(PYTHON) -m aetherforge.cli dashboard --port 8765

demo:
	bash scripts/capture_demo.sh nexus

data:
	$(PYTHON) -m aetherforge.cli data \
		-c configs/base.yaml \
		-c configs/domains/example_logistics.yaml \
		--output artifacts/data/logistics \
		-o data.synthetic.num_samples=128

validate:
	$(PYTHON) -m aetherforge.cli validate \
		-c configs/base.yaml \
		-c configs/domains/example_logistics.yaml

lint:
	$(PYTHON) -m ruff check aetherforge tests || true

clean:
	rm -rf artifacts/runs artifacts/data .pytest_cache
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true

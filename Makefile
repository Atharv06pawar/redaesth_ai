PYTHON ?= python

.PHONY: install bootstrap status research model-eval download test lint smoke pipeline train eval clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

bootstrap:
	$(PYTHON) -m redaesth_ai.cli bootstrap

status:
	$(PYTHON) -c "from pathlib import Path; print(Path('STATUS.md').read_text(encoding='utf-8'))"

research:
	$(PYTHON) -m redaesth_ai.cli research

model-eval:
	$(PYTHON) research/model_comparison/coaching_eval.py

download:
	$(PYTHON) pipeline/download.py

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

smoke:
	$(PYTHON) -m redaesth_ai.cli smoke-test

pipeline:
	$(PYTHON) pipeline/download.py

train:
	$(PYTHON) training/train.py --smoke-test

eval:
	$(PYTHON) evaluation/benchmarks/run_all_benchmarks.py

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; [shutil.rmtree(path, ignore_errors=True) for path in [Path('build'), Path('dist'), Path('.pytest_cache')]]"

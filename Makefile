PYTHON ?= python

.PHONY: install bootstrap status research model-eval download clean-dataset score-dataset build-final-dataset test lint smoke pipeline train eval clean

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

clean-dataset:
	$(PYTHON) pipeline/clean.py

score-dataset:
	$(PYTHON) pipeline/score.py

build-final-dataset:
	$(PYTHON) pipeline/build_final_dataset.py

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

smoke:
	$(PYTHON) -m redaesth_ai.cli smoke-test

pipeline:
	$(PYTHON) pipeline/download.py
	$(PYTHON) pipeline/clean.py
	$(PYTHON) pipeline/score.py
	$(PYTHON) pipeline/build_final_dataset.py

train:
	$(PYTHON) training/train.py --smoke-test

eval:
	$(PYTHON) evaluation/benchmarks/run_all_benchmarks.py

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; [shutil.rmtree(path, ignore_errors=True) for path in [Path('build'), Path('dist'), Path('.pytest_cache')]]"

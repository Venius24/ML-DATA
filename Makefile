PY ?= python

.PHONY: setup data eda cv errors final test all
setup:
	$(PY) -m venv .venv
	.venv/bin/pip install -r requirements.txt || .venv/Scripts/pip install -r requirements.txt
data:
	$(PY) scripts/download_data.py
eda:
	$(PY) scripts/eda.py
cv:
	$(PY) scripts/run_cv.py
errors:
	$(PY) scripts/error_analysis.py
final:
	$(PY) scripts/final_eval.py
test:
	$(PY) -m pytest -q
all: data eda cv errors final

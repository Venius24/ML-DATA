PY ?= python

.PHONY: setup data eda cv errors curve final summary cv2 final2 test all
setup:
	$(PY) -m venv .venv
	.venv/bin/python -m pip install -r requirements.txt || .venv/Scripts/python -m pip install -r requirements.txt
data:
	$(PY) scripts/download_data.py
eda:
	$(PY) scripts/eda.py
cv:
	$(PY) scripts/run_cv.py
errors:
	$(PY) scripts/error_analysis.py --model "lgbm raw-L2 (+geo) @600 trees"
	$(PY) scripts/error_analysis.py --out error_analysis_after_fix
curve:
	$(PY) scripts/learning_curve.py
final:
	$(PY) scripts/final_eval.py
summary:
	$(PY) scripts/summary.py
cv2:
	$(PY) scripts/run_cv_v2.py
final2:
	$(PY) scripts/final_eval_v2.py
test:
	$(PY) -m pytest -q
all:
	$(PY) scripts/run_all.py

# One command: `make demo` (setup if needed, all evals, then the web app on http://127.0.0.1:8000).
# Windows without make: `python scripts/setup.py` then `.venv\Scripts\python -m beatagent.cli demo`.
PYTHON ?= python3
ifeq ($(OS),Windows_NT)
  VPY := .venv/Scripts/python.exe
else
  VPY := .venv/bin/python
endif

.PHONY: setup demo evals evals-replay test lint check package serve testpack clean

setup:
	$(PYTHON) scripts/setup.py

demo: setup
	$(VPY) -m beatagent.cli demo

evals: setup
	$(VPY) -m beatagent.cli evals

evals-replay: setup
	$(VPY) -m beatagent.cli evals --replay

serve:
	$(VPY) -m beatagent.server

testpack:
	$(VPY) -m testpack.generate

clean:
	rm -rf runtime evals/out

test:
	$(VPY) -m pytest -q tests

lint:
	$(VPY) -m ruff check beatagent evals tests scripts testpack --line-length 120

check:
	$(VPY) scripts/check_claims.py

package:
	$(VPY) scripts/package_artifacts.py

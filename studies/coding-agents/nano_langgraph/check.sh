#!/usr/bin/env bash
# Typo-catcher for the LangGraph starters: syntax, import, offline tests.
# Needs the requirements.txt packages + pytest; no endpoint is contacted.
set -euo pipefail
cd "$(dirname "$0")"

echo "[1/3] syntax"
python -m py_compile lg_agent.py test_nano_langgraph.py

echo "[2/3] import"
python -c "import lg_agent"

echo "[3/3] tests"
python -m pytest -q

echo "OK"

#!/usr/bin/env bash
# Quick typo-catcher for nano_agent: syntax, import, then the offline tests.
# Needs only httpx + pytest; no model endpoint is contacted.
set -euo pipefail
cd "$(dirname "$0")"

echo "[1/3] syntax"
python -m py_compile nano_agent.py test_nano.py

echo "[2/3] import"
python -c "import nano_agent"

echo "[3/3] tests"
python -m pytest -q

echo "OK"

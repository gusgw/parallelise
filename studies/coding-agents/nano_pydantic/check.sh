#!/usr/bin/env bash
# Typo-catcher for the Pydantic-AI starter: syntax, import, offline tests.
# Needs pydantic-ai + pytest (see requirements.txt); no endpoint is contacted.
set -euo pipefail
cd "$(dirname "$0")"

echo "[1/3] syntax"
python -m py_compile pyd_agent.py test_nano_pydantic.py

echo "[2/3] import"
python -c "import pyd_agent"

echo "[3/3] tests"
python -m pytest -q

echo "OK"

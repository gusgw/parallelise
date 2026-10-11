#!/usr/bin/env bash
# Runs through `uv run` because this project needs its venv (deepagents, langgraph).
set -euo pipefail
cd "$(dirname "$0")"
export UV_LINK_MODE=copy

# --extra dev so pytest is installed INTO the project venv (otherwise `uv run
# pytest` runs an ephemeral pytest that can't see the project's deps).
echo "[1/3] py_compile (syntax)"
find src tests -name '*.py' -print0 | xargs -0 uv run --extra dev python -m py_compile

echo "[2/3] import-all smoke"
uv run --extra dev python -c "import importlib, pkgutil, agent_langgraph as p; [importlib.import_module(m.name) for m in pkgutil.walk_packages(p.__path__, p.__name__ + '.')]"

echo "[3/3] unit tests"
uv run --extra dev pytest -q

echo "OK"

#!/usr/bin/env bash
# Typo-catcher. eval_harness is stdlib + harness_common only, so it runs on the
# system Python with ../common/src on the path (no framework venv needed).
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="../common/src:src"

echo "[1/3] py_compile (syntax)"
find src tests -name '*.py' -print0 | xargs -0 python -m py_compile

echo "[2/3] import-all smoke"
python -c "import importlib, pkgutil, eval_harness as p; [importlib.import_module(m.name) for m in pkgutil.walk_packages(p.__path__, p.__name__ + '.')]"

echo "[3/3] unit tests"
python -m pytest -q

echo "OK"

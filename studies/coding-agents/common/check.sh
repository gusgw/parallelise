#!/usr/bin/env bash
# Fast typo-catcher: syntax -> imports -> unit tests.
# Steps 1 and 2 need only Python + httpx (no pytest), so they work on a
# bare interpreter before the dev tools are installed.
set -euo pipefail
cd "$(dirname "$0")"

echo "[1/3] py_compile (syntax)"
find src tests -name '*.py' -print0 | xargs -0 python -m py_compile

echo "[2/3] import-all smoke"
PYTHONPATH=src python -c "import importlib, pkgutil, harness_common as p; [importlib.import_module(m.name) for m in pkgutil.walk_packages(p.__path__, p.__name__ + '.')]"

echo "[3/3] unit tests"
PYTHONPATH=src python -m pytest -q

echo "OK"

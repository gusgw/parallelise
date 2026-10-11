#!/usr/bin/env bash
# Same typo-catcher as common/. Puts ../common/src and ../agent_min/src on the
# path so this runs without installing the path dependencies.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="../common/src:../agent_min/src:src"

echo "[1/3] py_compile (syntax)"
find src tests -name '*.py' -print0 | xargs -0 python -m py_compile

echo "[2/3] import-all smoke"
python -c "import importlib, pkgutil, agent_min_improved as p; [importlib.import_module(m.name) for m in pkgutil.walk_packages(p.__path__, p.__name__ + '.')]"

echo "[3/3] unit tests"
python -m pytest -q

echo "OK"

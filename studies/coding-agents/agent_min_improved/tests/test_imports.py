"""Import every agent_min_improved module — the cheapest typo catcher."""
import importlib
import pkgutil

import agent_min_improved


def test_import_all_modules():
    pkg = agent_min_improved
    for mod in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        importlib.import_module(mod.name)

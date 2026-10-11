"""Import every agent_min module — the cheapest typo catcher."""
import importlib
import pkgutil

import agent_min


def test_import_all_modules():
    for mod in pkgutil.walk_packages(agent_min.__path__, agent_min.__name__ + "."):
        importlib.import_module(mod.name)

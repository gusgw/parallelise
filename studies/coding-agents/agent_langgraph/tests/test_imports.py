"""Import every agent_langgraph module."""
import importlib
import pkgutil

import agent_langgraph


def test_import_all_modules():
    for mod in pkgutil.walk_packages(agent_langgraph.__path__, agent_langgraph.__name__ + "."):
        importlib.import_module(mod.name)

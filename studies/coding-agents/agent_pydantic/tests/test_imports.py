"""Import every agent_pydantic module."""
import importlib
import pkgutil

import agent_pydantic


def test_import_all_modules():
    for mod in pkgutil.walk_packages(agent_pydantic.__path__, agent_pydantic.__name__ + "."):
        importlib.import_module(mod.name)

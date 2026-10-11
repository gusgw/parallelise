"""Import every eval_harness module."""
import importlib
import pkgutil

import eval_harness


def test_import_all_modules():
    for mod in pkgutil.walk_packages(eval_harness.__path__, eval_harness.__name__ + "."):
        importlib.import_module(mod.name)

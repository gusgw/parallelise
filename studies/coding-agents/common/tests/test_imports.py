"""Import every harness_common module — the cheapest typo catcher."""
import importlib
import pkgutil

import harness_common


def test_import_all_modules():
    pkg = harness_common
    for mod in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        importlib.import_module(mod.name)

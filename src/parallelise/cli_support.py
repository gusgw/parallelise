"""Small helpers shared by the conversion commands and the study commands.

They live in a module of their own so that `parallelise.cli` and `parallelise.study.cli`
can both use them without importing each other.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from types import ModuleType
from typing import Any

from .config import Settings
from .errors import ParalleliseError


def lazy_import(name: str, hint: str = "") -> ModuleType:
    """Import parallelise.<name> on first use, turning an import failure into a plain error."""
    full = f"parallelise.{name}"
    try:
        return importlib.import_module(full)
    except ModuleNotFoundError as exc:
        if exc.name == full:
            raise ParalleliseError(
                f"The {name!r} part of parallelise is not available in this installation. "
                "Reinstall the package with 'pip install -e .' from the repository root."
            ) from exc
        raise ParalleliseError(
            f"The {name!r} part of parallelise needs the Python package {exc.name!r}, which is "
            f"not installed. {hint or 'Install it with pip.'}"
        ) from exc
    except ImportError as exc:
        raise ParalleliseError(
            f"The {name!r} part of parallelise could not be loaded ({exc}). "
            "Reinstall the package with 'pip install -e .'."
        ) from exc


def resolve_problem(problem: str, problems_dir: str) -> Path:
    """Accept either a path to a problem folder or the name of one under problems_dir."""
    direct = Path(problem)
    if direct.is_dir():
        return direct
    named = Path(problems_dir) / problem
    if named.is_dir():
        return named
    raise ParalleliseError(
        f"There is no problem folder {problem!r} or {str(named)!r}. Run from the repository "
        "root, or give the folder's path, or set --problems-dir."
    )


def settings_from_args(args: argparse.Namespace) -> Settings:
    """Build the settings from the environment, overridden by any options that were given."""
    overrides: dict[str, Any] = {
        "gateway_url": getattr(args, "gateway_url", None),
        "model": getattr(args, "model", None),
        "work_root": getattr(args, "work_root", None),
        "results_path": getattr(args, "results", None),
    }
    return Settings.from_env(overrides)

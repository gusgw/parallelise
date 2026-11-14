"""Logging set-up. Only the command line calls this; library code just logs."""

from __future__ import annotations

import logging
import sys

from .errors import ParalleliseError


def configure_logging(level: str = "INFO") -> None:
    """Send the package's log messages to standard error at ``level``.

    Calling it again replaces the earlier set-up.
    """
    numeric = logging.getLevelName(str(level).upper())
    if not isinstance(numeric, int):
        raise ParalleliseError(f"The log level {level!r} is not known. Use DEBUG, INFO, "
                               "WARNING or ERROR.")  # fmt: skip
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger = logging.getLogger("parallelise")
    logger.handlers = [handler]
    logger.setLevel(numeric)
    logger.propagate = False

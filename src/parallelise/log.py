"""Logging set-up. Only the command line calls this; library code just logs."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

from .errors import ParalleliseError


class _JsonFormatter(logging.Formatter):
    """Format each record as one JSON object on one line, for machine reading."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


def configure_logging(level: str = "INFO", json_lines: bool = False) -> None:
    """Send the package's log messages to standard error at ``level``, as plain lines or JSON.

    Calling it again replaces the earlier set-up.
    """
    numeric = logging.getLevelName(str(level).upper())
    if not isinstance(numeric, int):
        raise ParalleliseError(f"The log level {level!r} is not known. Use DEBUG, INFO, "
                               "WARNING or ERROR.")  # fmt: skip
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter() if json_lines else
                         logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger = logging.getLogger("parallelise")
    logger.handlers = [handler]
    logger.setLevel(numeric)
    logger.propagate = False

"""Logging set-up. Only the command-line layer calls this; library code just logs."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

from .errors import ConfigError

_HANDLER_MARK = "_parallelise_handler"


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
    """Send the package's log messages to standard error at the given level.

    Calling it again replaces the earlier set-up rather than adding a second handler.
    With json_lines true, each message is one JSON object per line.
    """
    numeric = logging.getLevelName(str(level).upper())
    if not isinstance(numeric, int):
        raise ConfigError(
            f"The log level {level!r} is not known. Use DEBUG, INFO, WARNING or ERROR."
        )
    logger = logging.getLogger("parallelise")
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_MARK, False):
            logger.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    setattr(handler, _HANDLER_MARK, True)
    if json_lines:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
    logger.addHandler(handler)
    logger.setLevel(numeric)
    logger.propagate = False

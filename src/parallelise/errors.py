"""Exceptions raised by the package.

Every error that a person can cause, whether by a wrong name, a wrong path, a file that
cannot be read or written, or a malformed data file, derives from ParalleliseError. The
command-line layer prints its message as one line instead of a stack trace, so each message
must say what went wrong and what to do next.
"""

from __future__ import annotations


class ParalleliseError(Exception):
    """Base class for every error this package raises on purpose."""


class ConfigError(ParalleliseError):
    """A setting is missing or has a value that cannot be used."""


class InputError(ParalleliseError, ValueError):
    """Something the user supplied (a name, a path, a folder or a file) cannot be used.

    It is also a ValueError, so that callers written before this class existed, which
    caught ValueError for a bad alias or a bad agent name, still work.
    """


class CheckError(ParalleliseError):
    """The check could not run at all (as opposed to running and failing a routine)."""


class AgentUnavailable(ParalleliseError):
    """An agent design cannot be used, usually because an optional extra is not installed."""


class ModelError(ParalleliseError):
    """The model endpoint could not be reached or returned an answer that cannot be used."""


class OracleError(ParalleliseError):
    """A notebook failed to run, or its ORACLE dictionary is missing or invalid."""


class ResultsError(ParalleliseError):
    """A run record or a price table is invalid, or a results file cannot be read or written."""

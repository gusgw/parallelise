"""Exceptions raised by the package.

Every error that the command-line layer should show to a person, rather than as a stack
trace, derives from ParalleliseError. Each message should say what went wrong and what to
do next.
"""

from __future__ import annotations


class ParalleliseError(Exception):
    """Base class for every error this package raises on purpose."""


class ConfigError(ParalleliseError):
    """A setting is missing or has a value that cannot be used."""


class CheckError(ParalleliseError):
    """The grading check could not run at all (as opposed to running and failing)."""


class AgentUnavailable(ParalleliseError):
    """An agent design cannot be used, usually because an optional extra is not installed."""


class ModelError(ParalleliseError):
    """The model endpoint could not be reached or returned an answer that cannot be used."""


class OracleError(ParalleliseError):
    """A notebook failed to run, or its ORACLE dictionary is missing or invalid."""


class ResultsError(ParalleliseError):
    """A run record is invalid or the results file cannot be read."""

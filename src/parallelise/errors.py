"""Exceptions raised by the package. Every error that a person can cause is a ParalleliseError, and
the command line prints its message as one line that says what went wrong and what to do next.
"""


class ParalleliseError(Exception):
    """Base class for every error this package raises on purpose."""


class AgentUnavailable(ParalleliseError):
    """An agent design cannot be used, usually because an optional extra is not installed."""


class ModelError(ParalleliseError):
    """The model endpoint could not be reached or returned an answer that cannot be used."""

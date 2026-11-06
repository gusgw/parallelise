"""Exceptions raised by the package.

Every error that a person can cause, by a wrong name, a wrong path, a setting that cannot be
used or a file that cannot be read or written, is a ParalleliseError. The command line prints
its message as one line instead of a stack trace, so each message says what went wrong and
what to do next.
"""


class ParalleliseError(Exception):
    """Base class for every error this package raises on purpose."""


class AgentUnavailable(ParalleliseError):
    """An agent design cannot be used, usually because an optional extra is not installed."""


class ModelError(ParalleliseError):
    """The model endpoint could not be reached or returned an answer that cannot be used."""

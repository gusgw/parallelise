"""Settings for the model endpoint and the working folders.

Each setting comes from an environment variable, and a command-line option, where there is one,
overrides it. One table gives, for each setting, its variable, its kind, and the test that a
usable value passes.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from .errors import ParalleliseError

#: A variable whose name contains one of these words is taken to hold a secret.
_SECRET_WORDS = ("KEY", "TOKEN", "SECRET", "PASSWORD")
#: Every variable that configures this program starts with this prefix.
_OWN_PREFIX = "PARALLELISE_"


def _web_address(value: str) -> bool:
    """True for an http or https address; a port that is not a number raises ValueError."""
    _ = urlsplit(value).port
    return value.startswith(("http://", "https://"))


#: Each setting: its variable, its kind, the test a usable value passes, and what that asks for.
_TABLE: dict[str, tuple[str, type, Callable[[Any], bool], str]] = {
    "gateway_url": ("PARALLELISE_GATEWAY_URL", str, _web_address,
                    "a web address that starts with http:// or https://"),
    "model": ("PARALLELISE_MODEL", str, bool, "the name of a model"),
    "temperature": ("PARALLELISE_TEMPERATURE", float, lambda v: 0 <= v <= 2,
                    "a number from 0 to 2"),
    "api_key": ("PARALLELISE_API_KEY", str, str.isascii,
                "plain letters, digits and punctuation, which an HTTP header can carry"),
    "request_timeout_s": ("PARALLELISE_TIMEOUT", float, lambda v: v > 0,
                          "a number of seconds above zero"),
    "time_limit_s": ("PARALLELISE_TIME_LIMIT", float, lambda v: v > 0,
                     "a number of seconds above zero"),
    "throttle_wait_s": ("PARALLELISE_THROTTLE_WAIT", float, lambda v: v >= 0,
                        "a number of seconds, or 0 never to wait"),
    "work_root": ("PARALLELISE_WORK_ROOT", str, bool, "the path of a folder"),
    "results_path": ("PARALLELISE_RESULTS", str, bool, "the path of a file"),
}  # fmt: skip


@dataclass
class Settings:
    """Everything that decides which model is called and where files are kept."""

    gateway_url: str = "http://localhost:4000/v1"
    model: str = "local"
    temperature: float = 0.2
    api_key: str = ""
    request_timeout_s: float = 600.0
    time_limit_s: float = 1800.0  # the wall-clock time an agent may spend on one run
    throttle_wait_s: float = 120.0  # the most waiting for one request refused with HTTP 429
    work_root: str = "work"
    results_path: str = "results/local-runs.jsonl"

    @classmethod
    def from_env(cls, overrides: Mapping[str, Any] | None = None,
                 environ: Mapping[str, str] | None = None) -> Settings:
        """Build settings from the environment and then ``overrides``, checking each value.

        An override of None is ignored, so that options that were not given can be passed on.
        """
        env = os.environ if environ is None else environ
        settings = cls()
        for name, (variable, kind, usable, wanted) in _TABLE.items():
            raw = (overrides or {}).get(name)
            if raw is None:
                raw = env.get(variable) or None
            if raw is None:
                continue
            try:
                value = kind(raw)
                ok = usable(value)
            except (TypeError, ValueError):
                ok = False
            if not ok:
                raise ParalleliseError(f"The value {raw!r} of {variable}, or of its option, "
                                       f"must be {wanted}.")  # fmt: skip
            setattr(settings, name, value.rstrip("/") if name == "gateway_url" else value)
        return settings


def without_secrets(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return a copy of the environment that is safe to give to code the model wrote.

    It leaves out every variable whose name holds KEY, TOKEN, SECRET or PASSWORD, in any
    capitals, and every variable of this program's own (PARALLELISE_), among them the gateway
    key, so that a compiled routine cannot read the key from its environment.
    """
    source = os.environ if environ is None else environ
    return {name: value for name, value in source.items()
            if not name.upper().startswith(_OWN_PREFIX)
            and not any(word in name.upper() for word in _SECRET_WORDS)}  # fmt: skip

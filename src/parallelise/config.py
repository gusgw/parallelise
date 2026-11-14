"""Settings for the model endpoint and the working folders.

Each setting comes from an environment variable, and a command-line option, where there is one,
overrides it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

#: A variable whose name contains one of these words is taken to hold a secret.
_SECRET_WORDS = ("KEY", "TOKEN", "SECRET", "PASSWORD")
#: Every variable that configures this program starts with this prefix.
_OWN_PREFIX = "PARALLELISE_"
#: The environment variable of each setting.
_VARIABLES = {
    "gateway_url": "PARALLELISE_GATEWAY_URL", "model": "PARALLELISE_MODEL",
    "temperature": "PARALLELISE_TEMPERATURE", "api_key": "PARALLELISE_API_KEY",
    "request_timeout_s": "PARALLELISE_TIMEOUT", "time_limit_s": "PARALLELISE_TIME_LIMIT",
    "throttle_wait_s": "PARALLELISE_THROTTLE_WAIT", "work_root": "PARALLELISE_WORK_ROOT",
    "results_path": "PARALLELISE_RESULTS",
}


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
        """Build settings from the environment and then ``overrides``.

        An override of None is ignored, so that options that were not given can be passed on.
        """
        env = os.environ if environ is None else environ
        settings = cls()
        for name, variable in _VARIABLES.items():
            raw = (overrides or {}).get(name)
            if raw is None:
                raw = env.get(variable) or None
            if raw is not None:
                setattr(settings, name, type(getattr(settings, name))(raw))
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

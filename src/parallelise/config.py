"""Settings for the model endpoint and the working folders.

Settings come from environment variables, with optional overrides (for example from
command-line options). The older MODEL_* variable names from the first version of this
work are still accepted, so existing scripts keep working.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any
from urllib.parse import urlsplit

from .errors import ConfigError

DEFAULT_GATEWAY_URL = "http://localhost:4000/v1"

#: A variable whose name contains one of these words is taken to hold a secret.
_SECRET_WORDS = ("KEY", "TOKEN", "SECRET", "PASSWORD")
#: Every variable that configures this program starts with this prefix.
_OWN_PREFIX = "PARALLELISE_"

# field name -> (current variable name, older fallback name or None)
_ENV_NAMES: dict[str, tuple[str, str | None]] = {
    "gateway_url": ("PARALLELISE_GATEWAY_URL", "MODEL_BASE_URL"),
    "model": ("PARALLELISE_MODEL", "MODEL_NAME"),
    "temperature": ("PARALLELISE_TEMPERATURE", "MODEL_TEMPERATURE"),
    "api_key": ("PARALLELISE_API_KEY", "MODEL_API_KEY"),
    "request_timeout_s": ("PARALLELISE_TIMEOUT", None),
    "time_limit_s": ("PARALLELISE_TIME_LIMIT", None),
    "throttle_wait_s": ("PARALLELISE_THROTTLE_WAIT", None),
    "work_root": ("PARALLELISE_WORK_ROOT", None),
    "results_path": ("PARALLELISE_RESULTS", None),
}


@dataclass
class Settings:
    """Everything that decides which model is called and where files are kept."""

    gateway_url: str = DEFAULT_GATEWAY_URL
    model: str = "local"
    temperature: float = 0.2
    api_key: str = ""
    request_timeout_s: float = 600.0
    #: Seconds of wall-clock time an agent may spend on one run before it is stopped.
    time_limit_s: float = 1800.0
    #: Most seconds to spend waiting, in total, for one model request that is refused with
    #: HTTP 429 (too many requests). Zero means do not wait.
    throttle_wait_s: float = 120.0
    work_root: str = "work"
    results_path: str = "results/runs.jsonl"

    def validate(self) -> Settings:
        """Raise ConfigError with a plain message if any value is unusable; return self."""
        if not isinstance(self.gateway_url, str) or not self.gateway_url.startswith(
            ("http://", "https://")
        ):
            raise ConfigError(
                f"The gateway URL {self.gateway_url!r} must start with http:// or https://. "
                "Set PARALLELISE_GATEWAY_URL, for example to http://localhost:4000/v1."
            )
        try:
            urlsplit(self.gateway_url).port  # noqa: B018 - reading it checks that it is a port
        except ValueError as exc:
            raise ConfigError(
                f"The gateway URL {self.gateway_url!r} is not a usable web address ({exc}). "
                "Set PARALLELISE_GATEWAY_URL, for example to http://localhost:4000/v1."
            ) from exc
        if not self.model:
            raise ConfigError("The model name is empty. Set PARALLELISE_MODEL or pass --model.")
        if not 0.0 <= self.temperature <= 2.0:
            raise ConfigError(
                f"The temperature {self.temperature} must lie between 0 and 2. "
                "Set PARALLELISE_TEMPERATURE to a value in that range."
            )
        if not self.request_timeout_s > 0:
            raise ConfigError(
                f"The request timeout {self.request_timeout_s} must be positive. "
                "Set PARALLELISE_TIMEOUT to a number of seconds above zero."
            )
        if not self.time_limit_s > 0:
            raise ConfigError(
                f"The time limit {self.time_limit_s} must be positive. Set PARALLELISE_TIME_LIMIT "
                "or pass --time-limit with a number of seconds above zero."
            )
        if not self.throttle_wait_s >= 0:
            raise ConfigError(
                f"The throttle wait {self.throttle_wait_s} must not be negative. Set "
                "PARALLELISE_THROTTLE_WAIT to a number of seconds, or to 0 to never wait."
            )
        if not self.api_key.isascii():
            raise ConfigError(
                "The API key holds characters that cannot be sent in an HTTP header. Check "
                "PARALLELISE_API_KEY: a key is made of plain letters, digits and punctuation."
            )
        return self

    @classmethod
    def from_env(
        cls,
        overrides: Mapping[str, Any] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> Settings:
        """Build validated settings from the environment, then apply overrides.

        An override whose value is None is ignored, so that unset command-line options can
        be passed through without special handling. The environ argument exists so that
        tests need not change the real environment.
        """
        env = os.environ if environ is None else environ
        values: dict[str, Any] = {}
        for name, (current, older) in _ENV_NAMES.items():
            raw = env.get(current)
            if raw is None and older is not None:
                raw = env.get(older)
            if raw is None or raw == "":
                continue
            values[name] = raw
        known = {f.name for f in fields(cls)}
        for name, value in (overrides or {}).items():
            if name not in known:
                raise ConfigError(
                    f"Unknown setting {name!r}; the known settings are {sorted(known)}."
                )
            if value is not None:
                values[name] = value
        for name in ("temperature", "request_timeout_s", "time_limit_s", "throttle_wait_s"):
            if name in values:
                try:
                    values[name] = float(values[name])
                except (TypeError, ValueError):
                    var = _ENV_NAMES[name][0]
                    raise ConfigError(
                        f"The value {values[name]!r} for {name} is not a number. "
                        f"Check {var} or the matching option."
                    ) from None
        if "gateway_url" in values:
            values["gateway_url"] = str(values["gateway_url"]).rstrip("/")
        for name in ("model", "api_key", "work_root", "results_path"):
            if name in values:
                values[name] = str(values[name])
        return cls(**values).validate()


def without_secrets(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return a copy of the environment that is safe to give to code the model wrote.

    The copy leaves out every variable whose name contains KEY, TOKEN, SECRET or PASSWORD
    (in any capitals), and every variable whose name begins with PARALLELISE_, which holds
    this program's own settings, among them the gateway key. A program that the agent
    compiled and runs can then not read the key from its environment. The argument defaults
    to the real environment.
    """
    source = os.environ if environ is None else environ
    return {
        name: value
        for name, value in source.items()
        if not name.upper().startswith(_OWN_PREFIX)
        and not any(word in name.upper() for word in _SECRET_WORDS)
    }

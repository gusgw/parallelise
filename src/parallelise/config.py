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

from .errors import ConfigError

DEFAULT_GATEWAY_URL = "http://localhost:4000/v1"

# field name -> (current variable name, older fallback name or None)
_ENV_NAMES: dict[str, tuple[str, str | None]] = {
    "gateway_url": ("PARALLELISE_GATEWAY_URL", "MODEL_BASE_URL"),
    "model": ("PARALLELISE_MODEL", "MODEL_NAME"),
    "temperature": ("PARALLELISE_TEMPERATURE", "MODEL_TEMPERATURE"),
    "api_key": ("PARALLELISE_API_KEY", "MODEL_API_KEY"),
    "request_timeout_s": ("PARALLELISE_TIMEOUT", None),
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
        for name in ("temperature", "request_timeout_s"):
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

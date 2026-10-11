"""The record of one graded run, and the file that holds all of them.

The store is a JSON Lines file: one JSON object per line. Appending a line is the only
way it changes, so an interrupted run cannot damage earlier records.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .errors import ResultsError

log = logging.getLogger("parallelise.results")

FAILURE_STAGES = (
    "none",
    "no-files",
    "tampered",
    "build",
    "test-one-worker",
    "test-many-workers",
    "benchmark",
    "timeout",
    "agent-error",
)


def now_iso() -> str:
    """Return the current time in UTC as an ISO 8601 string."""
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class RunRecord:
    """Everything worth knowing about one agent run and how the gate judged it."""

    problem: str
    backend: str
    model: str
    agent: str
    result: str  # "PASS" or "FAIL"
    failure_stage: str = "none"
    when: str = field(default_factory=now_iso)
    agent_version: str = ""  # short hash of the agent's source code
    masked: bool = False
    speedup_best: float | None = None
    wall_time_s: float = 0.0
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_mean_s: float = 0.0
    tokens_per_second: float = 0.0
    serving: str = "not recorded"
    quantisation: str = "not disclosed"
    notes: str = ""

    def validate(self) -> None:
        """Raise ResultsError if the result or failure stage is not one of the known values."""
        if self.result not in ("PASS", "FAIL"):
            raise ResultsError(f"The result {self.result!r} must be PASS or FAIL.")
        if self.failure_stage not in FAILURE_STAGES:
            raise ResultsError(
                f"The failure stage {self.failure_stage!r} is not one of "
                f"{', '.join(FAILURE_STAGES)}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary, ready for JSON."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunRecord:
        """Build a record from a dictionary, ignoring keys this version does not know."""
        names = {f.name for f in fields(cls)}
        try:
            return cls(**{k: v for k, v in data.items() if k in names})
        except TypeError as exc:
            raise ResultsError(f"A run record is missing a required field: {exc}.") from exc


class ResultsStore:
    """Append-only JSON Lines file of RunRecord objects."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: RunRecord) -> None:
        """Validate the record and add it as one line, creating the file if needed."""
        record.validate()
        line = json.dumps(record.to_dict(), sort_keys=False) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # One write call in append mode keeps concurrent runs from interleaving lines.
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line)
        log.info(
            "recorded %s %s %s in %s", record.problem, record.backend, record.result, self.path
        )

    def load(self) -> list[RunRecord]:
        """Read every record. A missing file gives an empty list; a damaged line is an error."""
        if not self.path.exists():
            return []
        records: list[RunRecord] = []
        with open(self.path, encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ResultsError(
                        f"Line {number} of {self.path} is not valid JSON ({exc.msg}). "
                        "Remove or repair that line."
                    ) from exc
                if not isinstance(data, dict):
                    raise ResultsError(f"Line {number} of {self.path} is not a JSON object.")
                records.append(RunRecord.from_dict(data))
        return records

"""The record of one graded run, and the file that holds all of them.

The store is a JSON Lines file: one JSON object per line. Appending a line is the only
way it changes, so an interrupted run cannot damage earlier records. A record gains fields
as the program learns to note more about a run; every new field has a default, so that
records written earlier still load, with the new fields marked as not recorded (None, an
empty text or an empty mapping).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .bench_check import SPEED_VERDICTS
from .errors import ResultsError

log = logging.getLogger("parallelise.results")

# "agent-error" is found only in records made before an agent that raised an error was judged
# by the check like any other; a new run never writes it, but it stays valid so that the old
# records load and validate.
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
    """Everything worth knowing about one agent run and how the check judged it.

    ``result`` is "PASS" when the check found the routine correct and keeping the rules,
    whatever its speed, and ``speed_verdict`` says how fast it was. Records made before the
    speed verdict was kept have None there; under the rule that applied then, a correct
    routine that missed the speed-up floor was recorded as a failure at the benchmark stage.
    """

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
    # Added later, so each of these is absent (None, "" or an empty mapping) in older records.
    temperature: float | None = None  # the sampling temperature the model was asked to use
    step_budget: int | None = None  # the most model steps the agent was allowed
    time_limit_s: float | None = None  # the wall-clock limit on the agent, in seconds
    agent_time_s: float | None = None  # seconds the agent ran, until it stopped
    check_time_s: float | None = None  # seconds the check took, after the agent stopped
    framework_versions: dict[str, str] = field(default_factory=dict)  # package -> version
    cpu_count: int | None = None  # processor cores the run could use
    package_version: str = ""  # the version of parallelise itself
    source_hash: str = ""  # short hash of the source files, so that any code change shows
    speed_verdict: str | None = None  # one of bench_check.SPEED_VERDICTS

    def validate(self) -> None:
        """Raise ResultsError if the result, failure stage or speed verdict is not a known value."""
        if self.result not in ("PASS", "FAIL"):
            raise ResultsError(f"The result {self.result!r} must be PASS or FAIL.")
        if self.speed_verdict is not None and self.speed_verdict not in SPEED_VERDICTS:
            raise ResultsError(
                f"The speed verdict {self.speed_verdict!r} is not one of "
                f"{', '.join(SPEED_VERDICTS)}."
            )
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
        known = {k: v for k, v in data.items() if k in names}
        if known.get("framework_versions") is None:
            known.pop("framework_versions", None)
        try:
            return cls(**known)
        except TypeError as exc:
            raise ResultsError(f"A run record is missing a required field: {exc}.") from exc


class ResultsStore:
    """Append-only JSON Lines file of RunRecord objects."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def ensure_writable(self) -> None:
        """Raise ResultsError if a record could not be added to the file.

        A run can take an hour, so this is called before it starts: finding out afterwards that
        the record cannot be saved would waste it. Nothing is created or changed.
        """
        if self.path.is_dir():
            raise ResultsError(
                f"The results file {self.path} is a folder. Give --results (or "
                "PARALLELISE_RESULTS) the path of a file."
            )
        if self.path.exists():
            if not os.access(self.path, os.W_OK):
                raise ResultsError(
                    f"The results file {self.path} cannot be written to. Check its permissions, "
                    "or give --results (or PARALLELISE_RESULTS) another path."
                )
            return
        folder = self.path.parent
        while not folder.exists() and folder != folder.parent:
            folder = folder.parent
        if not folder.is_dir():
            raise ResultsError(
                f"The results file {self.path} cannot be made, because {folder} is not a folder. "
                "Give --results (or PARALLELISE_RESULTS) another path."
            )
        if not os.access(folder, os.W_OK | os.X_OK):
            raise ResultsError(
                f"The results file {self.path} cannot be made, because the folder {folder} "
                "cannot be written to. Check its permissions, or give --results (or "
                "PARALLELISE_RESULTS) another path."
            )

    def append(self, record: RunRecord) -> None:
        """Validate the record and add it as one line, creating the file if needed."""
        record.validate()
        line = json.dumps(record.to_dict(), sort_keys=False) + "\n"
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # One write call in append mode keeps concurrent runs from interleaving lines.
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line)
        except OSError as exc:
            raise ResultsError(
                f"The record could not be added to {self.path} ({exc.strerror or exc}). Check "
                "the path and its permissions, or give --results (or PARALLELISE_RESULTS) "
                "another path."
            ) from exc
        log.info(
            "recorded %s %s %s in %s", record.problem, record.backend, record.result, self.path
        )

    def load(self) -> list[RunRecord]:
        """Read every record. A missing file gives an empty list; a damaged line is an error."""
        if not self.path.exists():
            return []
        records: list[RunRecord] = []
        try:
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
                        raise ResultsError(
                            f"Line {number} of {self.path} is not a JSON object."
                        )
                    records.append(RunRecord.from_dict(data))
        except UnicodeDecodeError as exc:
            raise ResultsError(
                f"The results file {self.path} is not a text file in UTF-8 form. Give --results "
                "(or PARALLELISE_RESULTS) the path of a results file."
            ) from exc
        except OSError as exc:
            raise ResultsError(
                f"The results file {self.path} cannot be read ({exc.strerror or exc}). Check "
                "the path and its permissions."
            ) from exc
        return records

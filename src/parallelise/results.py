"""The record of one run, and the file that holds all of them.

The store is a JSON Lines file: one JSON object per line. Appending a line is the only way it
changes, so an interrupted run cannot damage earlier records.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path

from .errors import ParalleliseError


def now_iso() -> str:
    """Return the current time in UTC as an ISO 8601 string."""
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class RunRecord:
    """Everything worth knowing about one agent run and how the judgement found its routine.

    ``correct`` says whether the routine was correct, whatever its speed; ``faster`` whether a
    correct routine beat the serial routine by the judgement's rule. ``repeat`` numbers the
    runs of one problem, backend, model and design, so that every repeat is counted.
    """

    problem: str
    backend: str
    model: str
    agent: str
    correct: bool
    failure_stage: str = "none"
    speedup: float | None = None  # the median serial time over the median time of the routine
    faster: bool = False
    scaling: float | None = None  # the routine's median time at one worker over all of them
    parallel: bool | None = None  # whether all its workers measurably beat one
    masked: bool = False
    repeat: int = 1
    serving: str = "not recorded"  # who served the model, such as its provider
    quantisation: str = "not disclosed"  # the numeric precision of the model's weights
    temperature: float = 0.0
    step_budget: int = 0
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_mean_s: float = 0.0
    tokens_per_second: float = 0.0
    agent_time_s: float = 0.0
    cores: int = 0  # the processor cores the judgement used as workers
    source_hash: str = ""  # identifies the program's own code
    serial_hash: str = ""  # identifies the serial routine that judged the run
    work_folder: str = ""  # holds the routine and the judgement's log
    notes: str = ""
    when: str = field(default_factory=now_iso)


class ResultsStore:
    """An append-only JSON Lines file of RunRecord objects."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def ensure_writable(self) -> None:
        """Raise ParalleliseError now if a record could not be added later, at the end of a run."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8"):
                pass
        except OSError as exc:
            raise ParalleliseError(
                f"Runs cannot be recorded in {self.path} ({exc.strerror or exc}). Give --results "
                "a file that can be written."
            ) from exc

    def append(self, record: RunRecord) -> None:
        """Add the record as one line; a single write in append mode keeps lines whole."""
        self.ensure_writable()
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record)) + "\n")

    def load(self) -> list[RunRecord]:
        """Read every record; a missing file gives none, and a line that is not one is an error."""
        if not self.path.exists():
            return []
        names = {f.name for f in fields(RunRecord)}
        records = []
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                records.append(RunRecord(**{k: v for k, v in data.items() if k in names}))
            except (ValueError, TypeError, AttributeError) as exc:
                raise ParalleliseError(
                    f"Line {number} of {self.path} is not a run record of this version ({exc})."
                ) from exc
        return records

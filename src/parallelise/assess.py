"""The backend-assessment stage: which parallel method suits this algorithm, and why?

The whole notebook, text and code, is given to a model, which is asked for a critical
analysis: the dependency structure, whether the work is bound by arithmetic or by memory,
what communication a distributed version would need, and a recommendation among OpenMP,
MPI, CUDA, a combination of them, or well-optimised serial code when the algorithm resists
parallelisation. For a complex algorithm the model may split the work into parts and
recommend a backend for each. The reply ends with a JSON verdict, which this module
extracts even if the model wraps it in prose or a code fence.

The first version of this stage was a stand-alone script, used in July 2026. Its prompt is
kept here with a few clarifications and the addition of the per-part list.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .errors import ParalleliseError
from .model import ModelClient

log = logging.getLogger("parallelise.assess")

SINGLE_BACKENDS = ("openmp", "mpi", "cuda", "serial")
COMBINATION = "combination"
RECOMMENDATIONS = SINGLE_BACKENDS + (COMBINATION,)

PROMPT = """You are a parallel-computing specialist. Below is a research Jupyter \
notebook implementing a numerical algorithm. It is being ported to compiled, parallel \
code, and you must recommend HOW to parallelise it.

Analyse:
1. Dependency structure: which loop iterations are independent? Is the algorithm \
embarrassingly parallel, a reduction, a stencil with neighbour dependencies, or \
inherently sequential?
2. Arithmetic intensity: per byte of data touched, how much computation is done? Is the \
algorithm compute-bound or memory-bandwidth-bound at realistic sizes? What does that \
imply for the achievable speed-up on shared-memory CPU, distributed CPU, and GPU?
3. Communication: if the work were split across distributed processes, what data would \
they need to exchange, and how often?
4. If the algorithm has separable parts with different characters, describe each part \
separately.

Then recommend among: OpenMP (shared-memory threads), MPI (distributed processes), \
CUDA (GPU), a combination of these, or SERIAL with good compiler optimisation if \
parallelisation would not pay. Be critical: name backends that are a poor fit and say why.

End your reply with exactly one JSON object on its own line, with these fields:
{"backend": "<openmp|mpi|cuda|serial|combination>", \
"parts": [{"part": "<name of the part>", "backend": "<openmp|mpi|cuda|serial>"}], \
"dependency_structure": "<one phrase>", "bound_by": "<compute|memory|latency|mixed>", \
"communication": "<one phrase>", "also_suitable": [...], "poor_fit": [...], \
"expected_speedup": "<one phrase>", "justification": "<at most 3 sentences>"}
Use "parts" only when the backend is "combination", and then list every part with its own \
single backend; otherwise leave "parts" as an empty list. The lists "also_suitable" and \
"poor_fit" hold backend names.

The notebook:
"""


@dataclass
class Part:
    """One separable part of an algorithm and the backend recommended for it."""

    part: str
    backend: str


@dataclass
class Verdict:
    """The model's recommendation for one notebook, with the analysis behind it."""

    backend: str
    parts: list[Part] = field(default_factory=list)
    dependency_structure: str = ""
    bound_by: str = ""
    communication: str = ""
    also_suitable: list[str] = field(default_factory=list)
    poor_fit: list[str] = field(default_factory=list)
    expected_speedup: str = ""
    justification: str = ""
    analysis: str = ""
    model: str = ""
    notebook: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0

    @property
    def chosen_backend(self) -> str:
        """The single backend to run: the recommendation, or for a combination the first
        part that is not serial (serial if every part is serial)."""
        if self.backend != COMBINATION:
            return self.backend
        for part in self.parts:
            if part.backend != "serial":
                return part.backend
        return "serial"

    def summary_line(self) -> str:
        """One plain sentence naming the recommendation."""
        if self.backend == COMBINATION:
            listed = "; ".join(f"{p.part}: {p.backend}" for p in self.parts)
            return f"A combination is recommended ({listed})."
        return f"The recommended backend is {self.backend}."

    def to_dict(self) -> dict[str, Any]:
        """The verdict as plain data, ready to be written as JSON."""
        return asdict(self)


def notebook_text(notebook: Path) -> str:
    """Return the notebook's cells, markdown and code, as one labelled text."""
    try:
        data = json.loads(notebook.read_text(encoding="utf-8"))
        cells = data.get("cells", [])
    except OSError as exc:
        raise ParalleliseError(
            f"The notebook {notebook} cannot be read ({exc.strerror or exc}). Check the path."
        ) from exc
    except (ValueError, AttributeError) as exc:
        raise ParalleliseError(
            f"The file {notebook} is not a Jupyter notebook (it is not valid JSON). "
            "Give the path of an .ipynb file."
        ) from exc
    parts = []
    for index, cell in enumerate(cells):
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        parts.append(f"--- cell {index} ({cell.get('cell_type', '?')}) ---\n{source}")
    return "\n\n".join(parts)


def _text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return [] if value in (None, "") else [str(value)]


def _verdict_from_object(obj: dict[str, Any]) -> Verdict | None:
    """Turn a decoded JSON object into a Verdict, or None if it is not a valid verdict."""
    raw = obj.get("backend", obj.get("primary"))  # "primary" is the July 2026 field name
    if not isinstance(raw, str) or raw.strip().lower() not in RECOMMENDATIONS:
        return None
    backend = raw.strip().lower()
    parts: list[Part] = []
    raw_parts = obj.get("parts") or []
    if isinstance(raw_parts, list):
        for item in raw_parts:
            if isinstance(item, dict) and str(item.get("backend", "")).lower() in SINGLE_BACKENDS:
                parts.append(Part(str(item.get("part", "")), str(item["backend"]).lower()))
    if backend == COMBINATION and not parts:
        return None  # a combination with no parts says nothing that can be acted on
    return Verdict(
        backend=backend,
        parts=parts if backend == COMBINATION else [],
        dependency_structure=str(obj.get("dependency_structure", "")),
        bound_by=str(obj.get("bound_by", "")),
        communication=str(obj.get("communication", "")),
        also_suitable=_text_list(obj.get("also_suitable")),
        poor_fit=_text_list(obj.get("poor_fit")),
        expected_speedup=str(obj.get("expected_speedup", "")),
        justification=str(obj.get("justification", "")),
    )


def parse_verdict(reply: str) -> Verdict | None:
    """Find the last valid verdict object in a reply, whatever surrounds it.

    Every opening brace is tried as the start of a JSON object, so a code fence, prose
    before or after, and nested lists inside the object are all handled.
    """
    decoder = json.JSONDecoder()
    found: Verdict | None = None
    position = reply.find("{")
    while position != -1:
        try:
            obj, end = decoder.raw_decode(reply, position)
        except ValueError:
            position = reply.find("{", position + 1)
            continue
        if isinstance(obj, dict):
            candidate = _verdict_from_object(obj)
            if candidate is not None:
                found = candidate
        position = reply.find("{", end)
    return found


def assess_notebook(notebook: Path, client: ModelClient) -> Verdict:
    """Ask the model which backend suits the notebook and return its verdict.

    The call is timed and counted by the client like every other call. Raises
    ParalleliseError if the notebook cannot be read or the reply holds no usable verdict.
    """
    text = notebook_text(Path(notebook))
    log.info("assessing %s with %s", Path(notebook).name, client.settings.model)
    result = client.chat([{"role": "user", "content": PROMPT + text}])
    reply = str(result.message.get("content") or "")
    verdict = parse_verdict(reply)
    if verdict is None:
        log.warning("no verdict in the reply: %r", reply[-300:])
        raise ParalleliseError(
            "The model's reply did not end with a usable JSON verdict, so no backend can be "
            "recommended. Try again, or try a stronger model with --model."
        )
    verdict.analysis = reply.strip()
    verdict.model = client.settings.model
    verdict.notebook = Path(notebook).name
    verdict.prompt_tokens = result.prompt_tokens
    verdict.completion_tokens = result.completion_tokens
    verdict.latency_s = round(result.latency_s, 3)
    return verdict


def write_verdict(verdict: Verdict, path: Path) -> None:
    """Write the verdict, with the full analysis text, as a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(verdict.to_dict(), indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")

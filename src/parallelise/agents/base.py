"""Shared definitions for the three agent designs.

The command-line layer and the agents agree on the result type and on the call
signature defined here. The system prompt is also built here, so that all three designs
give the model exactly the same instructions and the comparison between them is fair.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import httpx

    from parallelise.config import Settings
    from parallelise.model import ModelClient

from parallelise.errors import AgentUnavailable

__all__ = [
    "AgentResult",
    "AgentUnavailable",
    "Agent",
    "LatencyRecorder",
    "build_system_prompt",
    "check_feedback",
]


@dataclass
class AgentResult:
    """What an agent reports when its run ends.

    ``finished`` is true only when the checks passed at the end of the run. ``steps`` counts
    turns of the agent's own loop, ``requests`` counts calls made to the model, and the token
    counts and latencies are summed or listed over those calls.
    """

    finished: bool
    steps: int
    requests: int
    prompt_tokens: int
    completion_tokens: int
    latencies_s: list[float] = field(default_factory=list)
    notes: str = ""


class Agent(Protocol):
    """The interface every agent design implements."""

    name: str

    def run(
        self,
        work_dir: Path,
        task: str,
        settings: Settings,
        client: ModelClient,
        step_budget: int = 80,
    ) -> AgentResult:
        """Work in ``work_dir`` on ``task`` until the checks pass or the budget is spent."""
        ...


class LatencyRecorder:
    """Collects the time each model request takes, for the two framework agents.

    The framework agents make their own HTTP calls, so the shared model client cannot time
    them. This object is attached to their HTTP clients as request and response hooks. The
    time recorded runs from sending the request to receiving the response headers, which for
    a non-streaming call is the whole generation time.
    """

    def __init__(self) -> None:
        self.latencies_s: list[float] = []

    def _start(self, request: httpx.Request) -> None:
        request.extensions["parallelise_started"] = time.perf_counter()

    def _finish(self, response: httpx.Response) -> None:
        started = response.request.extensions.get("parallelise_started")
        if started is not None:
            self.latencies_s.append(time.perf_counter() - started)

    def sync_hooks(self) -> dict[str, list]:
        """Event hooks for an ``httpx.Client``."""

        def on_request(request: httpx.Request) -> None:
            self._start(request)

        def on_response(response: httpx.Response) -> None:
            self._finish(response)

        return {"request": [on_request], "response": [on_response]}

    def async_hooks(self) -> dict[str, list]:
        """Event hooks for an ``httpx.AsyncClient``."""

        async def on_request(request: httpx.Request) -> None:
            self._start(request)

        async def on_response(response: httpx.Response) -> None:
            self._finish(response)

        return {"request": [on_request], "response": [on_response]}


_BACKEND_GUIDANCE = {
    "openmp": (
        "Target OpenMP. Parallelise the hot loops with '#pragma omp parallel for', using "
        "'reduction(+:...)' for sums and taking care with shared writes. The Makefile builds "
        "with gcc -fopenmp."
    ),
    "mpi": (
        "Target MPI. The kernel is collective: assume MPI is already initialised in main(), "
        "and do NOT call MPI_Init or MPI_Finalize inside the kernel. Decompose the work "
        "across ranks so that each rank computes only its own share (a block of loop "
        "iterations, output rows or grid rows: lo = total*rank/size, hi = "
        "total*(rank+1)/size). Then combine the shares with the right MPI operation for this "
        "algorithm: MPI_Allreduce when every rank forms a partial sum of the same result; "
        "MPI_Allgather(v) when each rank produces a distinct slice of the output that must "
        "end up on all ranks; an MPI_Sendrecv halo exchange when each step needs the "
        "neighbouring ranks' boundary values. In the test, call MPI_Init in main, make every "
        "rank call the kernel the same number of times (otherwise mpirun hangs), let only "
        "rank 0 assert against reference.h within REF_TOL and print, then call "
        "MPI_Finalize. Build with mpicc and run the tests with mpirun -np 4."
    ),
    "cuda": (
        "Target CUDA. Write the kernel in the .cu file: a __global__ grid-stride kernel "
        "where each thread accumulates its share of the work, a shared-memory reduction "
        "within each block, and a host function with the exact signature from the header "
        '(declared extern "C") that launches the kernel, synchronises, checks errors '
        "(cudaGetLastError, cudaDeviceSynchronize), combines the per-block partial results "
        "and returns the answer. The test file is plain C, with no MPI and no OpenMP: its "
        "main() checks the result against reference.h within REF_TOL. Build with make, which "
        "uses nvcc. The trusted benchmark times one processor core against your GPU kernel, "
        "and the checks require a real GPU speed-up."
    ),
}

_NOT_KERNEL_HEADERS = {"reference.h", "problem.h", "testing.h"}


def _detect_backend(work_dir: Path) -> str:
    """Guess the backend from the work folder's Makefile; OpenMP when unsure."""
    try:
        makefile = (work_dir / "Makefile").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "openmp"
    if "nvcc" in makefile:
        return "cuda"
    if "mpicc" in makefile:
        return "mpi"
    return "openmp"


def _detect_kernel(work_dir: Path) -> str:
    """Name the kernel from the one header in the work folder that is not a helper."""
    headers = sorted(
        p.stem for p in work_dir.glob("*.h") if p.name not in _NOT_KERNEL_HEADERS
    )
    return headers[0] if headers else "kernel"


def build_system_prompt(work_dir: Path) -> str:
    """Return the instructions given to the model, identical for all three designs.

    The backend and the kernel name are read from the work folder, so the caller does not
    have to pass them separately.
    """
    backend = _detect_backend(work_dir)
    kernel = _detect_kernel(work_dir)
    source = kernel + (".cu" if backend == "cuda" else ".c")
    return (
        "You convert a research Jupyter notebook that implements a numerical algorithm into "
        "a parallel C kernel. Read the notebook with read_notebook to learn the algorithm. "
        f"Read {kernel}.h, which gives the exact signature to implement, and reference.h, "
        "which gives the expected values and REF_TOL. If problem.h is present, read it and "
        "use its shared helpers (the input fill and the checksum). Then write two files: "
        f"{source}, the parallel kernel that implements {kernel}.h, and test_{kernel}.c, a "
        "unit test that checks the kernel against reference.h within REF_TOL. A benchmark "
        "(bench.c) and its driver (bench_sweep.sh) are already provided: do not write or "
        "modify them. Split the work across the parallel workers so that each does only its "
        "own share, and combine the results appropriately; never make every worker repeat "
        "the whole computation. Write C that compiles without warnings, because the build "
        "treats warnings as errors: include every header whose functions you call (stdlib.h "
        "for malloc and free, math.h for fabs, string.h for memcpy); get the rank from "
        "MPI_Comm_rank and use rank == 0 for work that only rank 0 does (never compare "
        "MPI_COMM_WORLD to 0); guard edge ranks that have no neighbour with MPI_PROC_NULL in "
        "MPI_Sendrecv; and do not leave parameters unused. "
        + _BACKEND_GUIDANCE[backend]
        + " Build and run the tests with run_shell using 'make check', and keep fixing the "
        "code until it passes. Copy text exactly from what you have read, make small and "
        "exact edits, and finish only when the checks pass."
    )


def check_feedback(command: list[str], output: str) -> str:
    """Return the message that tells the model its checks are still failing."""
    return (
        "The checks are not passing yet, so you are not finished. Apply the fix by calling "
        "write_file (do not only describe it), then carry on. The output of `"
        + " ".join(command)
        + "` was:\n"
        + output
    )

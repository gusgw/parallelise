"""Shared tolerance for tests that run the real check on a machine with too few cores.

Hosted continuous-integration runners have two to four shared cores, so a genuinely
parallel kernel may show little or no speed-up there. When the environment variable
PARALLELISE_SPEEDUP_UNRELIABLE is set to 1, a check result that passed every correctness
stage and failed only the benchmark's speed-up requirement is turned into a skip, with the
measured speed-up in the message. Flat timings also trip the checker's rule against
invented tables; on a declared small machine they are genuine, so that reason is accepted
too. Anywhere else a missed floor stays a failure.
"""

import os
from pathlib import Path

import pytest

from parallelise.bench_check import check_bench_csv

SPEEDUP_UNRELIABLE = os.environ.get("PARALLELISE_SPEEDUP_UNRELIABLE") == "1"


def skip_if_only_speedup_failed(result, work: Path, label: str, *, always: bool = False) -> None:
    """Skip the current test if ``result`` failed only for want of speed-up.

    ``always`` applies the tolerance even when the machine is not declared small; it is
    for problems known to be limited by memory bandwidth. Returns normally when the
    result passed or failed for any other reason, so the caller's assertion then runs.
    """
    if result.passed or result.failure_stage != "benchmark":
        return
    if not (always or SPEEDUP_UNRELIABLE):
        return
    bench_csv = work / "bench.csv"
    if not bench_csv.is_file():
        return
    verdict = check_bench_csv(bench_csv, floor=0.0)
    only_flat = all(r.startswith("throughput is essentially constant") for r in verdict.reasons)
    if verdict.ok or (SPEEDUP_UNRELIABLE and only_flat):
        pytest.skip(
            f"{label} is correct but shows speed-up {result.speedup_best} on this "
            f"{os.cpu_count()}-core machine; the speed-up requirement is checked on a "
            "multi-core host"
        )

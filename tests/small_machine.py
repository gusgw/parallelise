"""Whether this machine's timings are judged.

Continuous integration runs on shared test servers with two to four busy processor cores,
where timings mean nothing. There the workflow sets PARALLELISE_JUDGE_SPEED to 0, and the
check then proves that the code builds, gives the right answers and runs its benchmark,
without judging the speed-up. Tests that are about the speed rules themselves are skipped
on such a machine; speed is judged on a real multi-core machine instead.
"""

import os

import pytest

SPEED_JUDGED = os.environ.get("PARALLELISE_JUDGE_SPEED", "1").strip() != "0"

needs_speed = pytest.mark.skipif(
    not SPEED_JUDGED,
    reason="speed is not judged on this machine (PARALLELISE_JUDGE_SPEED=0)",
)

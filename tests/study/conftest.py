"""Shared set-up for the study tests.

The study tests need no model, no network and no real problems. They use the small
synthetic problem in tests/fixtures/midpoint and plain dictionaries as run records.
"""

from pathlib import Path

import pytest


@pytest.fixture
def midpoint_fixture() -> Path:
    """Return the folder of the synthetic problem used by the masking tests."""
    return Path(__file__).parent.parent / "fixtures" / "midpoint"

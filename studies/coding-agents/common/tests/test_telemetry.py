"""Tests for harness_common.telemetry."""
import io
from contextlib import redirect_stdout

from harness_common import telemetry


def test_emit_then_parse_roundtrip():
    buf = io.StringIO()
    with redirect_stdout(buf):
        telemetry.emit(7, 12345)
    out = buf.getvalue()
    assert telemetry.parse(out) == {"turns": 7, "tokens": 12345}


def test_parse_takes_last_and_tolerates_noise():
    text = "hello\nTELEMETRY {\"turns\": 1, \"tokens\": 2}\nworld\nTELEMETRY {\"turns\": 9, \"tokens\": 99}\n"
    assert telemetry.parse(text) == {"turns": 9, "tokens": 99}


def test_parse_empty_or_garbled():
    assert telemetry.parse("") == {}
    assert telemetry.parse("no telemetry here") == {}
    assert telemetry.parse("TELEMETRY not-json") == {}

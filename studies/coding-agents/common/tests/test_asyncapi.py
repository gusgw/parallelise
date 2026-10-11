"""Tests for harness_common.asyncapi parsing/rendering (pure, no CLI)."""
from harness_common import asyncapi

SPEC = {
    "asyncapi": "3.0.0",
    "channels": {
        "Ch1": {"address": "mealworm.raw"},
        "Ch2": {"address": "mealworm.enriched"},
    },
    "operations": {
        "Op1": {"action": "receive", "channel": {"$ref": "#/channels/Ch1"}},
        "Op2": {"action": "send", "channel": {"$ref": "#/channels/Ch2"}},
    },
}


def test_parse_topology_channels():
    topo = asyncapi.parse_topology(SPEC)
    assert topo.channels == ["mealworm.enriched", "mealworm.raw"]


def test_parse_topology_operations():
    topo = asyncapi.parse_topology(SPEC)
    assert ("receive", "mealworm.raw") in topo.operations
    assert ("send", "mealworm.enriched") in topo.operations


def test_parse_topology_empty_spec():
    topo = asyncapi.parse_topology({})
    assert topo.channels == []
    assert topo.operations == []


def test_render_topology_uses_verbs():
    text = asyncapi.render_topology(asyncapi.parse_topology(SPEC))
    assert "subscribes mealworm.raw" in text
    assert "publishes mealworm.enriched" in text

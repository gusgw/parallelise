"""Tests for settings and logging set-up."""

import json
import logging

import pytest

from parallelise.config import DEFAULT_GATEWAY_URL, Settings
from parallelise.errors import ConfigError
from parallelise.log import configure_logging


def test_defaults_when_environment_is_empty():
    s = Settings.from_env(environ={})
    assert s.gateway_url == DEFAULT_GATEWAY_URL == "http://localhost:4000/v1"
    assert s.model == "local"
    assert s.temperature == 0.2
    assert s.request_timeout_s == 600
    assert s.work_root == "work"
    assert s.results_path == "results/runs.jsonl"


def test_current_names_beat_older_names():
    env = {"PARALLELISE_MODEL": "new", "MODEL_NAME": "old", "MODEL_API_KEY": "k"}
    s = Settings.from_env(environ=env)
    assert s.model == "new"
    assert s.api_key == "k"


def test_older_names_are_accepted():
    env = {"MODEL_BASE_URL": "http://example.test:8000/v1/", "MODEL_TEMPERATURE": "0.7"}
    s = Settings.from_env(environ=env)
    assert s.gateway_url == "http://example.test:8000/v1"
    assert s.temperature == 0.7


def test_overrides_win_and_none_is_ignored():
    env = {"PARALLELISE_MODEL": "x"}
    s = Settings.from_env({"model": "chosen", "gateway_url": None}, environ=env)
    assert s.model == "chosen"
    assert s.gateway_url == DEFAULT_GATEWAY_URL


@pytest.mark.parametrize(
    "env, words",
    [
        ({"PARALLELISE_GATEWAY_URL": "ftp://x"}, "http"),
        ({"PARALLELISE_TEMPERATURE": "3"}, "between 0 and 2"),
        ({"PARALLELISE_TEMPERATURE": "warm"}, "not a number"),
        ({"PARALLELISE_TIMEOUT": "0"}, "positive"),
    ],
)
def test_bad_values_give_plain_errors(env, words):
    with pytest.raises(ConfigError, match=words):
        Settings.from_env(environ=env)


def test_unknown_override_is_rejected():
    with pytest.raises(ConfigError):
        Settings.from_env({"colour": "red"}, environ={})


def test_configure_logging_is_idempotent_and_json(capsys):
    configure_logging("INFO", json_lines=True)
    configure_logging("INFO", json_lines=True)
    logger = logging.getLogger("parallelise.test")
    logger.info("hello")
    lines = [ln for ln in capsys.readouterr().err.splitlines() if ln]
    assert len(lines) == 1
    assert json.loads(lines[0])["message"] == "hello"
    configure_logging("WARNING")


def test_configure_logging_rejects_unknown_level():
    with pytest.raises(ConfigError):
        configure_logging("LOUD")

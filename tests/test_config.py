"""Tests for the settings, the environment given to compiled code, and the logging set-up."""

import json
import logging

import pytest

from parallelise.config import Settings, without_secrets
from parallelise.errors import ParalleliseError
from parallelise.log import configure_logging


def test_settings_come_from_the_environment_and_options_win():
    env = {"PARALLELISE_MODEL": "m", "PARALLELISE_TEMPERATURE": "0.7",
           "PARALLELISE_GATEWAY_URL": "http://host:4000/v1/", "PARALLELISE_TIME_LIMIT": "700"}
    settings = Settings.from_env({"model": "chosen", "time_limit_s": None}, environ=env)
    assert (settings.model, settings.temperature, settings.gateway_url, settings.time_limit_s) == (
        "chosen", 0.7, "http://host:4000/v1", 700.0)
    assert Settings.from_env(environ={}) == Settings()


@pytest.mark.parametrize(("variable", "value"), [
    ("PARALLELISE_GATEWAY_URL", "localhost:4000"), ("PARALLELISE_GATEWAY_URL", "http://h:port/v1"),
    ("PARALLELISE_TEMPERATURE", "warm"), ("PARALLELISE_TEMPERATURE", "3"),
    ("PARALLELISE_TIME_LIMIT", "0"), ("PARALLELISE_THROTTLE_WAIT", "-1"),
    ("PARALLELISE_API_KEY", "kéy"),
])  # fmt: skip
def test_a_value_that_cannot_be_used_names_its_variable(variable, value):
    with pytest.raises(ParalleliseError, match=variable):
        Settings.from_env(environ={variable: value})


def test_compiled_code_gets_no_secret_and_none_of_the_program_s_own_settings():
    env = {"PATH": "/bin", "OPENAI_API_KEY": "x", "MY_TOKEN": "y", "db_password": "z",
           "PARALLELISE_API_KEY": "w", "PARALLELISE_MODEL": "m", "HOME": "/home/u"}
    assert without_secrets(env) == {"PATH": "/bin", "HOME": "/home/u"}


def test_logging_can_be_set_up_twice_and_written_as_json(capsys):
    configure_logging("INFO")
    configure_logging("INFO", json_lines=True)
    logging.getLogger("parallelise.test").info("hello")
    (line,) = capsys.readouterr().err.splitlines()
    assert json.loads(line)["message"] == "hello"
    with pytest.raises(ParalleliseError, match="not known"):
        configure_logging("LOUD")

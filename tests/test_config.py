"""Tests for the settings and the environment given to compiled code."""

from parallelise.config import Settings, without_secrets


def test_settings_come_from_the_environment_and_options_win():
    env = {"PARALLELISE_MODEL": "ignored", "PARALLELISE_TEMPERATURE": "0.7",
           "PARALLELISE_TIMEOUT": "50"}
    settings = Settings.from_env({"model": "chosen", "time_limit_s": None}, environ=env)
    assert (settings.model, settings.temperature, settings.request_timeout_s,
            settings.time_limit_s) == ("chosen", 0.7, 50.0, 1800.0)
    assert Settings.from_env(environ={}) == Settings()


def test_compiled_code_gets_no_secret_and_none_of_the_program_s_own_settings():
    env = {"PATH": "/bin", "OPENAI_API_KEY": "x", "MY_TOKEN": "y", "db_password": "z",
           "PARALLELISE_API_KEY": "w", "PARALLELISE_MODEL": "m", "HOME": "/home/u"}
    assert without_secrets(env) == {"PATH": "/bin", "HOME": "/home/u"}

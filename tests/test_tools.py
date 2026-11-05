"""Tests for how the agent's tools run commands: the allow-list, the process group and the
environment. They need only make and, for one test, a C compiler."""

from __future__ import annotations

import os
import shutil
import signal
import time
from pathlib import Path

import pytest

from parallelise.agents import tools

pytestmark = pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")


def is_alive(pid: int) -> bool:
    """Say whether a process exists and is not merely waiting to be collected."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except OSError:
        return True
    return fields[0] != "Z"


@pytest.mark.parametrize(
    "command",
    [
        "python3 -c 'print(1)'",
        "python3",
        "bash script.sh",
        "sh script.sh",
        "bash -c hi",
        "sh -c hi",
        "/usr/bin/python3 -c 'print(1)'",
        "/bin/sh -c hi",
        "/bin/bash",
    ],
)
def test_interpreters_and_shells_are_refused(tmp_path, command):
    output = tools.run_shell(tmp_path, command)
    assert output.startswith("error: command not allowed"), output


def test_the_allow_list_has_no_interpreter_or_shell():
    assert not {"python3", "python", "bash", "sh"} & tools.ALLOWED_COMMANDS


def test_a_dot_slash_path_that_leaves_the_work_folder_is_refused(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    for command in (
        "./" + "../" * 20 + "usr/bin/python3 -c 'print(42)'",
        "./../outside",
    ):
        output = tools.run_shell(work, command)
        assert output.startswith("error: command not allowed"), output
        assert "42" not in output


def test_a_program_inside_the_work_folder_still_runs(tmp_path):
    compiler = shutil.which("gcc") or shutil.which("cc")
    if compiler is None:
        pytest.skip("no C compiler")
    (tmp_path / "hello.c").write_text(
        '#include <stdio.h>\nint main(void) { puts("hello"); return 0; }\n'
    )
    built = tools.run_shell(tmp_path, f"{Path(compiler).name} -o hello hello.c")
    assert built.startswith("exit 0"), built
    ran = tools.run_shell(tmp_path, "./hello")
    assert ran == "exit 0\nhello\n"


def test_make_still_runs_a_check_target(tmp_path):
    (tmp_path / "Makefile").write_text("check:\n\t@echo all good\n")
    assert tools.run_shell(tmp_path, "make check") == "exit 0\nall good\n"
    assert tools.run_check(tmp_path) == (True, "exit 0\nall good\n")


def test_mpirun_is_still_allowed(tmp_path):
    if shutil.which("mpirun") is None:
        pytest.skip("mpirun is not installed")
    assert tools.run_shell(tmp_path, "mpirun --version").startswith("exit 0")


def test_a_child_process_does_not_survive_a_timeout(tmp_path):
    # The recipe starts a background child that has closed its output, then waits. After
    # the timeout, neither make nor the child may still be running.
    (tmp_path / "Makefile").write_text(
        "hang:\n\tsleep 60 >/dev/null 2>&1 & echo $$! > child.pid; sleep 60\n"
    )
    started = time.monotonic()
    code, output = tools.run_process(tmp_path, ["make", "hang"], timeout_s=2)
    elapsed = time.monotonic() - started
    pid = int((tmp_path / "child.pid").read_text())
    try:
        assert code == -1 and "was stopped" in output
        assert elapsed < 20, "the command should be stopped soon after the timeout"
        for _ in range(50):  # allow a moment for the system to remove the process
            if not is_alive(pid):
                break
            time.sleep(0.1)
        assert not is_alive(pid), f"child process {pid} survived the timeout"
    finally:
        if is_alive(pid):
            os.kill(pid, signal.SIGKILL)


def test_a_child_holding_the_output_open_does_not_survive_either(tmp_path):
    (tmp_path / "Makefile").write_text("hang:\n\tsleep 60 & echo $$! > child.pid; sleep 60\n")
    code, _ = tools.run_process(tmp_path, ["make", "hang"], timeout_s=2)
    pid = int((tmp_path / "child.pid").read_text())
    try:
        assert code == -1
        for _ in range(50):
            if not is_alive(pid):
                break
            time.sleep(0.1)
        assert not is_alive(pid)
    finally:
        if is_alive(pid):
            os.kill(pid, signal.SIGKILL)


def test_secrets_are_not_visible_to_a_command(tmp_path, monkeypatch):
    monkeypatch.setenv("PARALLELISE_API_KEY", "gateway-secret-123")
    monkeypatch.setenv("PARALLELISE_MODEL", "some-model")
    monkeypatch.setenv("MODEL_API_KEY", "old-secret-789")
    monkeypatch.setenv("DEPLOY_TOKEN", "token-secret-456")
    monkeypatch.setenv("DB_PASSWORD", "password-secret-000")
    monkeypatch.setenv("HARMLESS_SETTING", "visible")
    (tmp_path / "Makefile").write_text(
        "show:\n"
        '\t@echo "key=[$$PARALLELISE_API_KEY] model=[$$PARALLELISE_MODEL] old=[$$MODEL_API_KEY]"\n'
        '\t@echo "token=[$$DEPLOY_TOKEN] password=[$$DB_PASSWORD]"\n'
        '\t@echo "harmless=[$$HARMLESS_SETTING] path_set=[$${PATH:+yes}]"\n'
    )
    output = tools.run_shell(tmp_path, "make show")
    assert "key=[] model=[] old=[]" in output
    assert "token=[] password=[]" in output
    assert "harmless=[visible] path_set=[yes]" in output
    assert "secret" not in output


def test_the_checking_command_also_gets_no_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("PARALLELISE_API_KEY", "gateway-secret-123")
    (tmp_path / "Makefile").write_text('check:\n\t@echo "key=[$$PARALLELISE_API_KEY]"\n')
    passed, output = tools.run_check(tmp_path)
    assert passed and "key=[]" in output and "secret" not in output


def test_benchmark_settings_do_not_reach_the_agents_commands(tmp_path, monkeypatch):
    """A caller's shell must not be able to switch off the agent's own speed requirement."""
    from parallelise.agents import tools

    monkeypatch.setenv("BENCH_FLOOR_ROLE", "report")
    monkeypatch.setenv("BENCH_JUDGE_SPEED", "0")
    (tmp_path / "Makefile").write_text(
        "show:\n\t@echo role=[$$BENCH_FLOOR_ROLE] judge=[$$BENCH_JUDGE_SPEED]\n"
    )
    out = tools.run_shell(tmp_path, "make -s show")
    assert "role=[]" in out and "judge=[]" in out, out

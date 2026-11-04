"""Tests for the check, using the synthetic OpenMP fixture and deliberately broken routines.

The check judges the agent's routine alone, in a fresh folder assembled from the problem. These
tests cover that judgement, the tamper rules, the log line about the agent's own test, and the
fresh folder itself.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from tests.small_machine import skip_if_only_speedup_failed

from parallelise import check as check_module
from parallelise.assemble import AssembleError, assemble_work_dir, file_digest
from parallelise.check import (
    FAILURE_STAGES,
    KEPT_FOLDER,
    _tampered_files,
    assemble_judging_folder,
    run_check,
)

FIXTURE = Path(__file__).parent / "fixtures" / "midpoint"


def have_openmp_compiler() -> bool:
    cc = shutil.which("gcc") or shutil.which("cc")
    if not cc or not shutil.which("make"):
        return False
    probe = subprocess.run(
        [cc, "-fopenmp", "-x", "c", "-", "-o", "/dev/null"],
        input="#include <omp.h>\nint main(void){return omp_get_max_threads()<0;}\n",
        capture_output=True, text=True, timeout=60, check=False,
    )  # fmt: skip
    return probe.returncode == 0


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not have_openmp_compiler(), reason="needs gcc with OpenMP and make"),
]


def folder(tmp_path: Path, routine_source: str | None = None) -> Path:
    work = assemble_work_dir(FIXTURE, "openmp", tmp_path / "w", include_reference=True)
    if routine_source is not None:
        (work / "midpoint.c").write_text(routine_source)
    return work


def check(work: Path, **kwargs):
    """Run the check on a work folder of the fixture problem for OpenMP."""
    return run_check(work, FIXTURE, "openmp", **kwargs)


HEADER = '#include <math.h>\n#include <omp.h>\n#include "midpoint.h"\n'
WRONG_ROUTINE = HEADER + "double midpoint(double a, double b, long n)\n" \
    "{ (void)a; (void)b; (void)n; return 1.0; }\n"  # fmt: skip


def test_failure_stage_names_match_the_contract():
    assert set(FAILURE_STAGES) == {
        "none", "no-files", "tampered", "build", "test-one-worker", "test-many-workers",
        "benchmark", "timeout",
    }  # fmt: skip


def test_reference_solution_passes(tmp_path):
    work = folder(tmp_path)
    result = check(work, timeout_s=240)
    skip_if_only_speedup_failed(result, work, "the fixture reference")
    assert result.passed, result.log
    assert result.failure_stage == "none"
    assert result.speedup_best and result.speedup_best > 1.3
    assert "bench stats OK" in result.log
    assert not (work / KEPT_FOLDER).exists(), "a passing check must delete the judging folder"
    assert not (work / "bench.csv").exists(), "the benchmark must run in the judging folder"


def test_missing_routine_is_no_files(tmp_path):
    work = folder(tmp_path)
    (work / "midpoint.c").unlink()
    result = check(work)
    assert (result.passed, result.failure_stage) == (False, "no-files")


def test_warning_is_a_build_failure(tmp_path):
    source = HEADER + "double midpoint(double a, double b, long n)\n" \
        "{ int unused; return (b - a) * (double)n * 0.0; }\n"  # fmt: skip
    result = check(folder(tmp_path, source))
    assert result.failure_stage == "build", result.log


def test_wrong_answer_fails_at_one_worker(tmp_path):
    result = check(folder(tmp_path, WRONG_ROUTINE))
    assert result.failure_stage == "test-one-worker", result.log


def test_answer_wrong_only_with_threads_fails_at_many_workers(tmp_path):
    reference = (FIXTURE / "openmp/reference/midpoint.c").read_text()
    broken = reference.replace(
        "return integral_sum * h;",
        "return omp_get_max_threads() > 1 ? 1.0 : integral_sum * h;",
    ).replace("#include <math.h>", "#include <math.h>\n#include <omp.h>")
    result = check(folder(tmp_path, broken))
    assert result.failure_stage == "test-many-workers", result.log


def test_serial_routine_fails_the_benchmark(tmp_path):
    reference = (FIXTURE / "openmp/reference/midpoint.c").read_text()
    serial = reference.replace("#pragma omp parallel for reduction(+ : integral_sum)\n", "")
    assert serial != reference
    result = check(folder(tmp_path, serial), timeout_s=240)
    assert result.failure_stage == "benchmark", result.log
    assert "BENCH CHECK FAIL" in result.log


def test_stricter_floor_than_the_makefile_fails(tmp_path):
    result = check(folder(tmp_path), timeout_s=240, speedup_floor=1000.0)
    assert result.failure_stage == "benchmark"
    assert not result.passed


def test_hanging_test_is_a_timeout(tmp_path):
    source = HEADER + "double midpoint(double a, double b, long n)\n" \
        "{ (void)a; (void)b; (void)n; for (;;) {} return 0.0; }\n"  # fmt: skip
    result = check(folder(tmp_path, source), timeout_s=5)
    assert result.failure_stage == "timeout", result.log
    assert "was not run, because the time limit of the check had been reached" in result.log


# --- the agent's own test program does not decide the verdict -------------------------------

EMPTY_TEST = "int main(void) { return 0; }\n"
ALWAYS_PASSING_TEST = (
    '#include <stdio.h>\nint main(void) { printf("12 checks, 0 failed\\n"); return 0; }\n'
)


@pytest.mark.parametrize("agent_test", [EMPTY_TEST, ALWAYS_PASSING_TEST], ids=["empty", "faked"])
def test_agent_test_that_asserts_nothing_cannot_save_a_wrong_routine(tmp_path, agent_test):
    work = folder(tmp_path, WRONG_ROUTINE)
    (work / "test_midpoint.c").write_text(agent_test)
    result = check(work)
    assert (result.passed, result.failure_stage) == (False, "test-one-worker"), result.log
    assert "midpoint matches the oracle: got 1 want" in result.log  # the problem's own test
    assert "built and passed at one worker" in result.log  # the agent's own test was fooled


def test_a_wrong_routine_fails_even_with_no_test_file_at_all(tmp_path):
    work = folder(tmp_path, WRONG_ROUTINE)
    (work / "test_midpoint.c").unlink()
    result = check(work)
    assert result.failure_stage == "test-one-worker", result.log
    assert "test_midpoint.c does not exist, so it was not run" in result.log


def test_agent_test_result_is_logged_and_never_changes_the_verdict(tmp_path):
    cases = {
        "failed": ("int main(void) { return 1; }\n", "built but failed at one worker"),
        "did not build": ("this is not C\n", "did not build"),
        "does not exist": (None, "does not exist, so it was not run"),
    }
    for number, (agent_test, wording) in enumerate(cases.values()):
        work = assemble_work_dir(
            FIXTURE, "openmp", tmp_path / f"w{number}", include_reference=True
        )
        test_file = work / "test_midpoint.c"
        if agent_test is None:
            test_file.unlink()
        else:
            test_file.write_text(agent_test)
        result = check(work, timeout_s=240)
        skip_if_only_speedup_failed(result, work, "the fixture reference")
        assert result.passed, result.log  # the routine is right, so the verdict is a pass
        assert wording in result.log
        assert "no effect on the verdict" in result.log


def test_passing_agent_test_is_logged(tmp_path):
    work = folder(tmp_path)
    result = check(work, timeout_s=240)
    skip_if_only_speedup_failed(result, work, "the fixture reference")
    assert "test_midpoint.c built and passed at one worker" in result.log


def test_hanging_agent_test_is_logged_and_does_not_change_the_verdict(tmp_path, monkeypatch):
    monkeypatch.setattr(check_module, "AGENT_TEST_LIMIT_S", 2)
    work = folder(tmp_path)
    (work / "test_midpoint.c").write_text("int main(void) { for (;;) {} }\n")
    result = check(work, timeout_s=240)
    skip_if_only_speedup_failed(result, work, "the fixture reference")
    assert result.passed, result.log
    assert "did not finish before the time limit" in result.log


def test_the_agents_own_test_is_built_with_the_problems_makefile(tmp_path, monkeypatch):
    # With the tamper rules switched off, a work folder Makefile that cannot build anything
    # must still not matter: the agent's own test is built in a fresh folder.
    monkeypatch.setattr(check_module, "_tampered_files", lambda *args: [])
    work = folder(tmp_path)
    (work / "Makefile").write_text("all:\n\t@false\n")
    result = check(work, timeout_s=240)
    skip_if_only_speedup_failed(result, work, "the fixture reference")
    assert result.passed, result.log
    assert "test_midpoint.c built and passed at one worker" in result.log


# --- the judging folder: fresh, and kept only on failure ---------------------------------------


def test_failed_check_keeps_the_judging_folder_as_check_fail(tmp_path):
    work = folder(tmp_path, WRONG_ROUTINE)
    result = check(work)
    kept = work / KEPT_FOLDER
    assert result.failure_stage == "test-one-worker"
    assert kept.is_dir() and f"kept as {kept}" in result.log
    assert (kept / "midpoint.c").read_text() == WRONG_ROUTINE
    assert (kept / "test_midpoint.c").read_bytes() == (
        FIXTURE / "openmp/reference/test_midpoint.c"
    ).read_bytes()
    assert (kept / "Makefile").read_bytes() == (FIXTURE / "openmp/Makefile").read_bytes()
    assert (kept / "test_midpoint").is_file(), "the folder holds what was built and run"


def test_a_later_passing_check_removes_the_old_check_fail(tmp_path):
    work = folder(tmp_path, WRONG_ROUTINE)
    assert not check(work).passed and (work / KEPT_FOLDER).is_dir()
    shutil.copy2(FIXTURE / "openmp/reference/midpoint.c", work / "midpoint.c")
    result = check(work, timeout_s=240)
    skip_if_only_speedup_failed(result, work, "the fixture reference")
    assert result.passed, result.log
    assert not (work / KEPT_FOLDER).exists()


def test_the_judging_folder_takes_every_trusted_file_from_the_problem(tmp_path):
    work = folder(tmp_path)
    trusted = ["bench.c", "Makefile", "testing.h", "midpoint.h", "reference.h",
               "bench_sweep.sh", "check_bench.py"]  # fmt: skip
    for filename in trusted + ["test_midpoint.c"]:
        (work / filename).write_text("/* the agent rewrote this */\n")
    (work / "GNUmakefile").write_text("check:\n\t@true\n")
    (work / "helper.c").write_text("int helper;\n")

    judging = assemble_judging_folder(
        FIXTURE, "openmp", tmp_path / "judge", routine=work / "midpoint.c"
    )

    source = assemble_work_dir(FIXTURE, "openmp", tmp_path / "source")
    for filename in trusted:
        assert file_digest(judging / filename) == file_digest(source / filename), filename
    assert (judging / "test_midpoint.c").read_bytes() == (
        FIXTURE / "openmp/reference/test_midpoint.c"
    ).read_bytes()
    assert (judging / "midpoint.c").read_bytes() == (work / "midpoint.c").read_bytes()
    assert not (judging / "GNUmakefile").exists()
    assert not (judging / "helper.c").exists()
    assert {p.name for p in judging.iterdir()} == {p.name for p in source.iterdir()} | {
        "midpoint.c", "test_midpoint.c",
    }


def test_judging_folder_can_hold_the_agents_own_test_instead(tmp_path):
    mine = tmp_path / "mine.c"
    mine.write_text("int main(void) { return 0; }\n")
    judging = assemble_judging_folder(
        FIXTURE, "openmp", tmp_path / "judge",
        routine=FIXTURE / "openmp/reference/midpoint.c", test=mine,
    )  # fmt: skip
    assert (judging / "test_midpoint.c").read_text() == mine.read_text()


def test_verdict_comes_from_the_fresh_folder_even_if_tamper_detection_is_bypassed(
    tmp_path, monkeypatch
):
    # Switch the tamper rules off, then sabotage the work folder in every way an agent could:
    # a Makefile that passes everything, a benchmark and test that assert nothing, and a
    # GNUmakefile. The routine is wrong, so the verdict must still be a failure.
    monkeypatch.setattr(check_module, "_tampered_files", lambda *args: [])
    work = folder(tmp_path, WRONG_ROUTINE)
    (work / "Makefile").write_text("check:\n\t@true\n\ntest_midpoint bench:\n\t@true\n")
    (work / "bench.c").write_text("int main(void) { return 0; }\n")
    (work / "test_midpoint.c").write_text(EMPTY_TEST)
    (work / "GNUmakefile").write_text("check:\n\t@true\n")
    result = check(work)
    assert (result.passed, result.failure_stage) == (False, "test-one-worker"), result.log


def test_problem_without_a_known_good_test_cannot_be_checked(tmp_path):
    problem = tmp_path / "problem"
    shutil.copytree(FIXTURE, problem)
    (problem / "openmp/reference/test_midpoint.c").unlink()
    work = folder(tmp_path)
    with pytest.raises(AssembleError, match="known-good test"):
        run_check(work, problem, "openmp")


def test_unknown_backend_is_refused_plainly(tmp_path):
    with pytest.raises(AssembleError, match="is not one of"):
        run_check(folder(tmp_path), FIXTURE, "fortran")


def test_secrets_are_not_given_to_the_code_under_judgement(tmp_path, monkeypatch):
    monkeypatch.setenv("PARALLELISE_API_KEY", "gw-value-8a1f")
    monkeypatch.setenv("PARALLELISE_MODEL", "some-model")
    monkeypatch.setenv("SOME_SERVICE_TOKEN", "tk-value-4b2e")
    monkeypatch.setenv("HARMLESS_SETTING", "visible")
    spy = (
        "#include <stdio.h>\n#include <stdlib.h>\n"
        "__attribute__((constructor)) static void spy(void)\n"
        "{ const char *k = getenv(\"PARALLELISE_API_KEY\"), *t = getenv(\"SOME_SERVICE_TOKEN\"),"
        " *m = getenv(\"PARALLELISE_MODEL\"), *h = getenv(\"HARMLESS_SETTING\");\n"
        "  printf(\"[spy key=%s token=%s model=%s harmless=%s]\\n\", k ? k : \"absent\","
        " t ? t : \"absent\", m ? m : \"absent\", h ? h : \"absent\"); }\n"
    )
    result = check(folder(tmp_path, spy + WRONG_ROUTINE))
    assert result.failure_stage == "test-one-worker", result.log
    assert "[spy key=absent token=absent model=absent harmless=visible]" in result.log
    assert "gw-value-8a1f" not in result.log and "tk-value-4b2e" not in result.log


# --- the tamper stage --------------------------------------------------------------------------


def test_clean_folder_has_matching_digests(tmp_path):
    work = folder(tmp_path)
    manifest = json.loads((work / ".trusted.json").read_text())["sha256"]
    for name in ("Makefile", "bench.c", "testing.h", "midpoint.h", "check_bench.py"):
        assert name in manifest
    assert "midpoint.c" not in manifest
    fresh = assemble_work_dir(FIXTURE, "openmp", tmp_path / "fresh")
    assert _tampered_files(work, fresh, "midpoint") == []


@pytest.mark.parametrize("victim", ["bench.c", "Makefile", "testing.h", "midpoint.h"])
def test_edited_trusted_file_is_tampered(tmp_path, victim):
    work = folder(tmp_path)
    with (work / victim).open("a") as handle:
        handle.write("\n/* edited */\n")
    result = check(work, timeout_s=60)
    assert (result.passed, result.failure_stage) == (False, "tampered")
    assert f"{victim} has been changed" in result.log
    assert (work / KEPT_FOLDER).is_dir()


@pytest.mark.parametrize(
    "victim", ["bench.c", "Makefile", "testing.h", "midpoint.h", "reference.h", "check_bench.py"]
)
def test_rewriting_the_fingerprints_as_well_is_still_tampered(tmp_path, victim):
    work = folder(tmp_path)
    with (work / victim).open("a") as handle:
        handle.write("\n/* edited */\n")
    manifest_path = work / ".trusted.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sha256"][victim] = file_digest(work / victim)
    manifest_path.write_text(json.dumps(manifest))
    assert _tampered_files(work, assemble_work_dir(FIXTURE, "openmp", tmp_path / "f"), "midpoint")
    result = check(work, timeout_s=60)
    assert (result.passed, result.failure_stage) == (False, "tampered")
    assert f"{victim} differs from the copy made from the problem's own files" in result.log


@pytest.mark.parametrize("name", ["GNUmakefile", "makefile"])
def test_a_makefile_that_make_would_read_first_is_tampered(tmp_path, name):
    work = folder(tmp_path)
    (work / name).write_text("check:\n\t@echo my own recipe\n")
    result = check(work, timeout_s=60)
    assert (result.passed, result.failure_stage) == (False, "tampered")
    assert f"{name} is present" in result.log


def test_removed_trusted_file_is_tampered(tmp_path):
    work = folder(tmp_path)
    (work / "bench_sweep.sh").unlink()
    assert check(work, timeout_s=60).failure_stage == "tampered"


def test_missing_manifest_is_tampered(tmp_path):
    work = folder(tmp_path)
    (work / ".trusted.json").unlink()
    result = check(work, timeout_s=60)
    assert result.failure_stage == "tampered"


def test_trusted_files_are_judged_before_the_routine_is_looked_for(tmp_path):
    work = folder(tmp_path)
    (work / "midpoint.c").unlink()
    with (work / "bench.c").open("a") as handle:
        handle.write("\n/* edited */\n")
    assert check(work, timeout_s=60).failure_stage == "tampered"


def test_a_folder_assembled_elsewhere_is_not_tampered_for_its_package_path(tmp_path):
    # The two checker scripts name the place the package is installed, which differs from
    # one computer to the next. That alone must not count as tampering, but any other change
    # to them must.
    work = folder(tmp_path)
    script = work / "check_bench.py"
    lines = script.read_text().splitlines()
    moved = [
        "sys.path.insert(0, '/another/computer/src')" if line.startswith("sys.path.insert")
        else line
        for line in lines
    ]  # fmt: skip
    assert moved != lines
    fresh = assemble_work_dir(FIXTURE, "openmp", tmp_path / "fresh")
    manifest_path = work / ".trusted.json"

    def write(text: str) -> None:
        script.write_text(text)
        manifest = json.loads(manifest_path.read_text())
        manifest["sha256"]["check_bench.py"] = file_digest(script)
        manifest_path.write_text(json.dumps(manifest))

    write("\n".join(moved) + "\n")
    assert _tampered_files(work, fresh, "midpoint") == []
    write("\n".join([*moved, "print('always fine')"]) + "\n")
    assert _tampered_files(work, fresh, "midpoint") != []


def test_agent_tool_can_still_write_the_routine(tmp_path):
    from parallelise.agents.tools import write_file

    work = folder(tmp_path)
    write_file(work, "midpoint.c", WRONG_ROUTINE)
    fresh = assemble_work_dir(FIXTURE, "openmp", tmp_path / "fresh")
    assert _tampered_files(work, fresh, "midpoint") == []

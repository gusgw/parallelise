"""Tests for the rules about deleting folders.

parallelise never deletes a folder it did not make unless it is told to with --force. The one
exception is the work folder that `run` makes for a cell: a later run of the same cell may
replace it, but only if it carries the marker file that `run` wrote when it made it.
"""

import pytest
from parallelise.folders import (
    MARKER_NAME,
    has_marker,
    prepare_destination,
    prepare_own_folder,
    write_marker,
)
from tests.test_assemble import FIXTURE
from tests.test_cli import install_fakes, run_args

from parallelise import cli
from parallelise.assemble import AssembleError, assemble_work_dir
from parallelise.errors import ParalleliseError
from parallelise.results import ResultsStore


def holding_a_file(folder):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "precious.txt").write_text("keep me")
    return folder


# --- prepare_destination ---------------------------------------------------------------------


def test_a_new_or_empty_destination_is_accepted(tmp_path):
    prepare_destination(tmp_path / "new")
    (tmp_path / "empty").mkdir()
    prepare_destination(tmp_path / "empty")
    assert (tmp_path / "empty").is_dir()


def test_a_destination_with_files_is_refused_and_left_alone(tmp_path):
    folder = holding_a_file(tmp_path / "w")
    with pytest.raises(ParalleliseError) as raised:
        prepare_destination(folder)
    message = str(raised.value)
    assert str(folder) in message and "--force" in message
    assert (folder / "precious.txt").read_text() == "keep me"


def test_force_empties_the_destination(tmp_path):
    folder = holding_a_file(tmp_path / "w")
    prepare_destination(folder, force=True)
    assert not folder.exists() or not any(folder.iterdir())


def test_a_destination_that_is_a_file_is_refused_even_with_force(tmp_path):
    target = tmp_path / "afile"
    target.write_text("x")
    for force in (False, True):
        with pytest.raises(ParalleliseError, match="not a folder"):
            prepare_destination(target, force=force)
    assert target.read_text() == "x"


# --- the folders that run makes itself ---------------------------------------------------------


def test_a_marked_folder_is_replaced_without_force(tmp_path):
    folder = holding_a_file(tmp_path / "cell")
    write_marker(folder)
    assert has_marker(folder) and (folder / MARKER_NAME).is_file()
    prepare_own_folder(folder)
    assert not folder.exists()


def test_an_unmarked_folder_with_files_is_refused(tmp_path):
    folder = holding_a_file(tmp_path / "cell")
    with pytest.raises(ParalleliseError) as raised:
        prepare_own_folder(folder)
    assert str(folder) in str(raised.value) and "--force" in str(raised.value)
    assert (folder / "precious.txt").exists()


def test_force_replaces_an_unmarked_folder(tmp_path):
    folder = holding_a_file(tmp_path / "cell")
    prepare_own_folder(folder, force=True)
    assert not folder.exists()


def test_an_empty_or_absent_folder_needs_nothing(tmp_path):
    prepare_own_folder(tmp_path / "absent")
    (tmp_path / "empty").mkdir()
    prepare_own_folder(tmp_path / "empty")
    assert not (tmp_path / "empty").exists()


# --- assemble --------------------------------------------------------------------------------


def test_assemble_refuses_a_folder_that_holds_files(tmp_path):
    folder = holding_a_file(tmp_path / "w")
    with pytest.raises(ParalleliseError, match="--force"):
        assemble_work_dir(FIXTURE, "openmp", folder)
    assert (folder / "precious.txt").exists()


def test_assemble_with_force_replaces_the_folder(tmp_path):
    folder = holding_a_file(tmp_path / "w")
    assemble_work_dir(FIXTURE, "openmp", folder, force=True)
    assert not (folder / "precious.txt").exists() and (folder / "Makefile").is_file()


def test_assemble_accepts_an_empty_folder(tmp_path):
    (tmp_path / "w").mkdir()
    assemble_work_dir(FIXTURE, "openmp", tmp_path / "w")
    assert (tmp_path / "w" / "Makefile").is_file()


def test_assemble_command_refuses_then_obeys_force(tmp_path, capsys):
    folder = holding_a_file(tmp_path / "w")
    command = ["assemble", str(FIXTURE), "openmp", str(folder)]
    assert cli.main(command) == 1
    err = capsys.readouterr().err
    assert str(folder) in err and "--force" in err and "Traceback" not in err
    assert (folder / "precious.txt").exists()
    assert cli.main([*command, "--force"]) == 0
    assert not (folder / "precious.txt").exists()


def test_a_problem_with_no_backend_does_not_delete_the_destination(tmp_path):
    """The destination is judged only after the problem has been found usable."""
    folder = holding_a_file(tmp_path / "w")
    with pytest.raises(AssembleError):
        assemble_work_dir(FIXTURE, "mpi", folder, force=True)
    assert (folder / "precious.txt").exists()


# --- study mask ------------------------------------------------------------------------------


def test_mask_command_refuses_then_obeys_force(tmp_path, capsys):
    folder = holding_a_file(tmp_path / "masked")
    command = ["study", "mask", str(FIXTURE), "problem_a", str(folder)]
    assert cli.main(command) == 1
    err = capsys.readouterr().err
    assert str(folder) in err and "--force" in err and "Traceback" not in err
    assert (folder / "precious.txt").exists()
    assert cli.main([*command, "--force"]) == 0
    assert not (folder / "precious.txt").exists()
    assert (folder / "problem_a.ipynb").is_file()


def test_mask_into_an_empty_folder_needs_no_force(tmp_path):
    (tmp_path / "masked").mkdir()
    assert cli.main(["study", "mask", str(FIXTURE), "problem_a", str(tmp_path / "masked")]) == 0


# --- run -------------------------------------------------------------------------------------


def cell_folder(tmp_path):
    return tmp_path / "work" / "integrate" / "openmp" / "m1" / "basic"


def test_run_marks_its_work_folder_and_replaces_it_next_time(monkeypatch, tmp_path):
    calls = install_fakes(monkeypatch, tmp_path)
    assert cli.main(run_args(tmp_path)) == 0
    work = cell_folder(tmp_path)
    assert has_marker(work)
    (work / "left-over.txt").write_text("from the first run")
    assert cli.main(run_args(tmp_path)) == 0  # replaced, because the marker is there
    assert not (work / "left-over.txt").exists() and has_marker(work)
    assert calls["assembled"][2] == work


def test_run_refuses_a_work_folder_it_did_not_make(monkeypatch, tmp_path, capsys):
    calls = install_fakes(monkeypatch, tmp_path)
    work = holding_a_file(cell_folder(tmp_path))
    assert cli.main(run_args(tmp_path)) == 1
    err = capsys.readouterr().err
    assert str(work) in err and "--force" in err and "Traceback" not in err
    assert (work / "precious.txt").exists()
    assert "assembled" not in calls
    assert not (tmp_path / "runs.jsonl").exists()


def test_run_with_force_replaces_an_unmarked_work_folder(monkeypatch, tmp_path):
    install_fakes(monkeypatch, tmp_path)
    work = holding_a_file(cell_folder(tmp_path))
    assert cli.main([*run_args(tmp_path), "--force"]) == 0
    assert not (work / "precious.txt").exists() and has_marker(work)
    assert len(ResultsStore(tmp_path / "runs.jsonl").load()) == 1


def test_run_refuses_an_unmarked_masked_copy(monkeypatch, tmp_path, capsys):
    from types import SimpleNamespace

    install_fakes(monkeypatch, tmp_path, routine="problem_a")
    masked = holding_a_file(tmp_path / "work" / "_masked" / "integrate")
    real = cli._lazy
    monkeypatch.setattr(
        cli, "_lazy",
        lambda name, hint="": SimpleNamespace(mask_problem=lambda *a, **k: None)
        if name == "study.mask" else real(name),
    )
    assert cli.main([*run_args(tmp_path), "--masked", "problem_a"]) == 1
    err = capsys.readouterr().err
    assert str(masked) in err and "--force" in err
    assert (masked / "precious.txt").exists()

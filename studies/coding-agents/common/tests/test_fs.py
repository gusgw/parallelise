"""Tests for harness_common.tools.fs (confined filesystem ops)."""
import pytest

from harness_common.tools import fs


def test_safe_path_allows_subpath(tmp_path):
    p = fs.safe_path(str(tmp_path), "a/b.txt")
    assert str(p).startswith(str(tmp_path.resolve()))


def test_safe_path_allows_root(tmp_path):
    assert fs.safe_path(str(tmp_path), ".") == tmp_path.resolve()


def test_safe_path_rejects_parent_escape(tmp_path):
    with pytest.raises(ValueError):
        fs.safe_path(str(tmp_path), "../escape")


def test_safe_path_rejects_absolute(tmp_path):
    with pytest.raises(ValueError):
        fs.safe_path(str(tmp_path), "/etc/passwd")


def test_write_then_read(tmp_path):
    fs.write_file(str(tmp_path), "d/x.txt", "hello")
    assert fs.read_file(str(tmp_path), "d/x.txt") == "hello"


def test_read_file_truncates_large(tmp_path):
    fs.write_file(str(tmp_path), "big.txt", "x" * 20000)
    out = fs.read_file(str(tmp_path), "big.txt", max_chars=100)
    assert "truncated" in out
    assert len(out) < 300


def test_read_file_line_range(tmp_path):
    fs.write_file(str(tmp_path), "f.txt", "a\nb\nc\nd\ne")
    out = fs.read_file(str(tmp_path), "f.txt", start=2, count=2)
    assert "b\nc" in out
    assert "lines 2-3 of 5" in out


def test_edit_replaces_and_counts(tmp_path):
    fs.write_file(str(tmp_path), "x.txt", "a a a")
    msg = fs.edit_file(str(tmp_path), "x.txt", "a", "b")
    assert "3" in msg
    assert fs.read_file(str(tmp_path), "x.txt") == "b b b"


def test_edit_missing_old_raises(tmp_path):
    fs.write_file(str(tmp_path), "x.txt", "abc")
    with pytest.raises(ValueError):
        fs.edit_file(str(tmp_path), "x.txt", "zzz", "y")


def test_edit_error_quotes_real_text(tmp_path):
    # old has phantom indentation (the classic model mistake); the error must
    # quote the real, unindented file text so the model can copy it.
    fs.write_file(str(tmp_path), "x.py", "from m import (\n    a,\n    b,\n)\n")
    with pytest.raises(ValueError) as err:
        fs.edit_file(str(tmp_path), "x.py", "    from m import (\n    a,\n", "x")
    assert "actually contains" in str(err.value)
    assert "from m import (" in str(err.value)


def test_nearest_region_no_probe_or_no_hit():
    assert fs.nearest_region("text", "   \n  ") == ""          # blank old
    assert fs.nearest_region("aaa\nbbb", "zzz") == ""           # nothing similar


def test_ls_marks_dirs(tmp_path):
    fs.write_file(str(tmp_path), "f.txt", "1")
    (tmp_path / "sub").mkdir()
    entries = fs.ls(str(tmp_path), ".")
    assert "f.txt" in entries
    assert "sub/" in entries


def test_glob_finds_py(tmp_path):
    fs.write_file(str(tmp_path), "pkg/a.py", "x")
    fs.write_file(str(tmp_path), "pkg/b.txt", "x")
    assert fs.glob(str(tmp_path), "**/*.py") == ["pkg/a.py"]


def test_grep_finds_and_skips_git(tmp_path):
    fs.write_file(str(tmp_path), "code.py", "needle here")
    fs.write_file(str(tmp_path), ".git/config", "needle in git")
    hits = fs.grep(str(tmp_path), "needle")
    assert any("code.py" in h for h in hits)
    assert not any(".git" in h for h in hits)

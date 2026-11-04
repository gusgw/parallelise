"""Rules about writing into folders without destroying anything that matters.

parallelise never deletes a folder it did not make unless the user says so with ``--force``.
A destination that already holds files is refused with a message that names the folder and
the option. The one exception is the work folder that ``parallelise run`` makes for a cell, a
cell being one problem on one backend with one model and one agent design. A later run of
the same cell replaces its own earlier work folder, but only if the folder carries the marker
file that ``run`` wrote when it made it, so that a folder that merely happens to sit at the
same path is never deleted by mistake.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from .errors import InputError

log = logging.getLogger("parallelise.folders")

#: The file that ``parallelise run`` writes into a folder it makes, to show that it made it.
MARKER_NAME = ".parallelise-made"
_MARKER_TEXT = (
    "This folder was made by 'parallelise run'. A later run of the same cell may delete it "
    "and make it again.\n"
)


def _is_empty(folder: Path) -> bool:
    return not any(folder.iterdir())


def _refuse_non_folder(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise InputError(
            f"{path} is a file, not a folder. Choose a path where a folder can be made."
        )


def prepare_destination(dest: Path, force: bool = False) -> None:
    """Make ``dest`` ready to be written into: not there yet, or an empty folder.

    A folder that holds anything is refused unless ``force`` is true, when it is deleted. A
    path that is a file is always refused. Raises ``InputError`` with a plain message.
    """
    dest = Path(dest)
    _refuse_non_folder(dest)
    if not dest.exists():
        return
    if _is_empty(dest):
        return
    if not force:
        raise InputError(
            f"The folder {dest} already holds files, and parallelise will not delete them. "
            "Choose an empty or new folder, or add --force to delete what is there first."
        )
    log.warning("deleting %s because --force was given", dest)
    _delete(dest)


def write_marker(folder: Path) -> None:
    """Write the marker that shows that ``parallelise run`` made ``folder``."""
    (Path(folder) / MARKER_NAME).write_text(_MARKER_TEXT, encoding="utf-8")


def has_marker(folder: Path) -> bool:
    """Return whether ``folder`` carries the marker written by ``write_marker``."""
    return (Path(folder) / MARKER_NAME).is_file()


def prepare_own_folder(folder: Path, force: bool = False) -> None:
    """Clear the way for a folder that ``parallelise run`` is about to make.

    Nothing is done if the folder does not exist, and an empty folder is removed. A folder
    that carries the marker was made by an earlier run of the same cell, and is deleted. Any
    other folder that holds files is refused, with a message that names it and ``--force``,
    unless ``force`` is true. A path that is a file is always refused.
    """
    folder = Path(folder)
    _refuse_non_folder(folder)
    if not folder.exists():
        return
    if _is_empty(folder):
        folder.rmdir()
        return
    if has_marker(folder) or force:
        _delete(folder)
        return
    raise InputError(
        f"The folder {folder} was not made by 'parallelise run' (it has no {MARKER_NAME} "
        "marker file), so it will not be deleted. Move it away, choose another work root "
        "with --work-root, or add --force to delete it and start again."
    )


def _delete(folder: Path) -> None:
    try:
        shutil.rmtree(folder)
    except OSError as exc:
        raise InputError(
            f"The folder {folder} could not be deleted ({exc.strerror or exc}). Check its "
            "permissions, or choose another folder."
        ) from exc


def ensure_folder(folder: Path, what: str) -> None:
    """Make ``folder`` if it is not there, or raise ``InputError`` saying why it cannot be.

    ``what`` says what the folder is for, such as "the work root", to start the message.
    """
    folder = Path(folder)
    _refuse_non_folder(folder)
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise InputError(
            f"{what.capitalize()} {folder} cannot be made or used ({exc.strerror or exc}). "
            "Check the path and its permissions, or choose another one."
        ) from exc

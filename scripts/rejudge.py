"""Judge again, with the present judgement, the routines stored from the July and October runs.

Each stored routine was written against headers that no longer exist. Before judging, the
script changes a copy of the routine, and none of its logic: the include of its own header
becomes an include of the generated header, under the routine's old name; an include of
``problem.h`` is replaced by that file's text as it was published at commit 4e7dd1d (input
fills and checksums, which a routine has no need of); and an include of ``reference.h`` is
replaced by the old tolerance alone, so that a routine which reads an expected value fails to
build. The old contract for the MPI routines of scan, decay and reservoir left each block of
the result on its own process, and the present one wants the whole result on process 0, so
for those routines a gather is added after the routine's own code. The problem's serial
routine is renamed to match. A masked routine is judged under its alias, unless it defines the
declared name instead: the old masking renamed the files of cg but not its function cgsolve.

The changes need the repository's history and the judgement needs the compilers, so the work
is done in two steps, which may run on different machines:

    python scripts/rejudge.py prepare FOLDER [PROBLEM ...]
    python scripts/rejudge.py judge FOLDER RESULTS.jsonl [BACKEND ...]

The first writes each changed routine, its serial routine and a description into a folder of
its own under FOLDER. The second judges them in turn and appends one line of JSON for each to
RESULTS.jsonl; a routine already recorded there is not judged again, so a stopped run resumes.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from parallelise import judge, notebook

ROOT = Path(__file__).resolve().parent.parent
PUBLISHED = "4e7dd1d"
DISTRIBUTED = {"scan", "decay", "reservoir"}
GATHER = """
#include <mpi.h>
#include <stdlib.h>

/* Added by the re-judgement: the old contract left each block of the result on its own
 * process, and the present one wants the whole result on process 0. */
void {name}(const double *in, double *out, long n)
{{
    stored_{name}(in, out, n);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int *counts = malloc(sizeof(int) * (size_t)size), *starts = malloc(sizeof(int) * (size_t)size);
    for (int r = 0; r < size; r++) {{
        starts[r] = (int)(n * r / size);
        counts[r] = (int)(n * (r + 1) / size - starts[r]);
    }}
    MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : (void *)(out + starts[rank]), counts[rank],
                MPI_DOUBLE, out, counts, starts, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    free(counts);
    free(starts);
}}
"""


def old_file(problem: str, name: str) -> str:
    """Return a problem file as it was published, or an empty text if it did not exist."""
    shown = subprocess.run(["git", "show", f"{PUBLISHED}:problems/{problem}/{name}"],
                           cwd=ROOT, capture_output=True, text=True)  # fmt: skip
    return shown.stdout if shown.returncode == 0 else ""


def stored_routines() -> list[dict]:
    """Every stored routine, with its problem, backend, model, design and variant."""
    found = []
    for path in sorted((ROOT / "results" / "campaign-2026-10" / "runs").rglob("*.c*")):
        variant, agent, problem, backend, model = path.relative_to(ROOT).parts[3:8]
        found.append(dict(file=path, variant=variant, agent=agent, problem=problem,
                          backend=backend, model=model, study="october"))  # fmt: skip
    for path in sorted((ROOT / "results" / "runs").rglob("*.c*")):
        problem, backend, model, agent = path.relative_to(ROOT).parts[2:6]
        found.append(dict(file=path, variant="unmasked", agent=agent, problem=problem,
                          backend=backend, model=model, study="july"))  # fmt: skip
    # The routine is the file named after the problem, or after its alias in a masked run; the
    # others are the agents' own tests and trials.
    return [r for r in found
            if r["file"].stem == r["problem"] or re.fullmatch(r"kernel_[a-z]", r["file"].stem)]


def prepared(entry: dict, name: str, serial_name: str) -> tuple[str, str]:
    """Return the changed copy of the routine and the renamed serial routine."""
    text = entry["file"].read_text(encoding="utf-8", errors="replace")
    text = re.sub(rf'#include\s+"{re.escape(entry["file"].stem)}\.h"', f'#include "{name}.h"', text)
    helpers = old_file(entry["problem"], "problem.h")
    text = re.sub(r'#include\s+"problem\.h"', lambda _: helpers, text)
    tolerance = re.search(r"#define\s+REF_TOL\s+\S+", old_file(entry["problem"], "reference.h"))
    text = re.sub(r'#include\s+"reference\.h"', lambda _: tolerance.group(0) if tolerance else "",
                  text)  # fmt: skip
    if entry["backend"] == "mpi" and entry["problem"] in DISTRIBUTED:
        text, found = re.subn(rf"\bvoid\s+{re.escape(name)}\s*\(([^)]*)\)\s*\{{",
                              lambda m: f"static void stored_{name}({m.group(1)})\n{{", text,
                              count=1)  # fmt: skip
        if found:
            text += GATHER.format(name=name)
    serial = (ROOT / "problems" / entry["problem"] / "serial.c").read_text(encoding="utf-8")
    return text, re.sub(rf"\b{re.escape(serial_name)}\b", name, serial)


def prepare(folder: Path, wanted: set[str]) -> None:
    declarations: dict[str, str] = {}
    count = 0
    for number, entry in enumerate(stored_routines()):
        problem_dir = ROOT / "problems" / entry["problem"]
        if (wanted and entry["problem"] not in wanted) or not (problem_dir / "serial.c").is_file():
            continue
        if entry["problem"] not in declarations:
            nb = problem_dir / f"{entry['problem']}.ipynb"
            declarations[entry["problem"]] = notebook.declaration(nb).name
        serial_name = declarations[entry["problem"]]
        source = entry["file"].read_text(encoding="utf-8", errors="replace")
        alias = entry["file"].stem
        masked = entry["variant"] == "masked" and re.search(rf"\b{alias}\s*\(", source)
        name = alias if masked else serial_name
        routine, serial = prepared(entry, name, serial_name)
        place = folder / f"{number:04d}"
        place.mkdir(parents=True, exist_ok=True)
        suffix = ".cu" if entry["file"].suffix == ".cu" else ".c"
        (place / f"routine{suffix}").write_text(routine, encoding="utf-8")
        (place / "serial.c").write_text(serial, encoding="utf-8")
        described = {k: (str(v.relative_to(ROOT)) if isinstance(v, Path) else v)
                     for k, v in entry.items()}  # fmt: skip
        (place / "entry.json").write_text(json.dumps({**described, "name": name}))
        count += 1
    print(f"prepared {count} routines in {folder}")


def judge_all(folder: Path, out: Path, backends: set[str]) -> None:
    done = set()
    if out.is_file():
        done = {json.loads(line)["file"] for line in out.read_text().splitlines() if line.strip()}
    declarations = {}
    for place in sorted(p for p in folder.iterdir() if (p / "entry.json").is_file()):
        entry = json.loads((place / "entry.json").read_text())
        if entry["file"] in done or (backends and entry["backend"] not in backends):
            continue
        nb = ROOT / "problems" / entry["problem"] / f"{entry['problem']}.ipynb"
        if entry["problem"] not in declarations:
            declarations[entry["problem"]] = notebook.declaration(nb)
        routine = next(place.glob("routine.c*"))
        verdict = judge.judge(nb, declarations[entry["problem"]], routine, place / "serial.c",
                              entry["backend"], name=entry["name"])  # fmt: skip
        entry.update(correct=verdict.correct, stage=verdict.stage, speedup=verdict.speedup,
                     faster=verdict.faster, scaling=verdict.scaling, parallel=verdict.parallel,
                     why=verdict.log[-1] if verdict.log else "")  # fmt: skip
        with open(out, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        print(entry["file"], entry["stage"], entry["speedup"], flush=True)


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "prepare":
        prepare(Path(argv[1]), set(argv[2:]))
        return 0
    if len(argv) >= 3 and argv[0] == "judge":
        judge_all(Path(argv[1]), Path(argv[2]), set(argv[3:]))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

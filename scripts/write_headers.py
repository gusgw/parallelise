"""Write each problem's header beside its serial routine, as the harness generates it.

Every serial routine includes its problem's header, which the harness generates from the
notebook's declaration afresh for every run and judgement; it never reads these copies. They
are kept so that a reader or an editor can find the file that each serial routine includes.
Run this, as "make headers", after changing a declaration: a test fails until the copies match.
"""

from pathlib import Path

from parallelise import generate, notebook

for nb in sorted((Path(__file__).resolve().parent.parent / "problems").glob("*/*.ipynb")):
    decl = notebook.declaration(nb)
    (nb.parent / f"{decl.name}.h").write_text(generate.header(decl, decl.name), encoding="utf-8")
    print(f"wrote problems/{nb.parent.name}/{decl.name}.h")

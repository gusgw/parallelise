"""Jacobi demonstration: the notebook's heat solve, served by the parallel C.

Run from bindings/ after building:  .venv/bin/python demo_jacobi_use.py

Builds the notebook's initial grid, runs its pure-Python sweeps and the wrapped
parallel C kernel, and checks both final checksums against the notebook-derived
oracle within its tolerance.
"""
import json
import time

import numpy as np

import jacobi_par


def jacobi_python(u, iters):
    n = len(u)
    for _ in range(iters):
        un = [row[:] for row in u]
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                un[i][j] = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1])
        u = un
    return u


def main():
    oracle = json.load(open("../problems/jacobi/reference.json"))
    n = oracle["params"]["N"]
    iters = oracle["params"]["ITERS"]
    expected = oracle["expected"]["CHECKSUM"]
    tol = oracle["tol"]

    u0 = np.zeros((n, n))
    u0[0, :] = 1.0

    t0 = time.perf_counter()
    upy = jacobi_python(u0.tolist(), iters)
    t_py = time.perf_counter() - t0
    cs_py = sum(sum(row) for row in upy)

    t0 = time.perf_counter()
    uc = jacobi_par.jacobi(u0, iters)
    t_c = time.perf_counter() - t0

    print(f"n = {n}, sweeps = {iters}")
    print(f"python : checksum {cs_py:.10f}  in {t_py:8.4f} s")
    print(f"C+OpenMP: checksum {uc.sum():.10f}  in {t_c:8.4f} s   ({t_py / t_c:,.0f}x faster)")
    print(f"oracle : {expected:.10f}  (tol {tol})")
    ok = abs(cs_py - expected) < tol and abs(float(uc.sum()) - expected) < tol
    print("agreement:", "both OK" if ok else "MISMATCH")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

"""The end-goal demonstration: the notebook's algorithm, served by the parallel C.

Compares three ways of computing the same integral:
  1. the notebook's own pure-Python midpoint loop (what the researcher had),
  2. the parallel C kernel imported through nanobind (what the conversion produces),
and checks both agree with the notebook-derived oracle (reference.json) within its
tolerance. In a notebook the only change for the researcher is:

    from integrate_par import integrate      # instead of the Python def

Run from bindings/ after building:  python3 demo_notebook_use.py
"""
import json
import time

import integrate_par


def integrate_python(a, b, n):
    """The notebook's original pure-Python midpoint loop (for comparison)."""
    h = (b - a) / n
    s = 0.0
    for i in range(n):
        x = a + (i + 0.5) * h
        s += 4.0 / (1.0 + x * x)
    return s * h


def main():
    oracle = json.load(open("../problems/integrate/reference.json"))
    n = oracle["params"]["N_TEST"]
    expected = oracle["expected"]["EXPECTED"]
    tol = oracle["tol"]

    t0 = time.perf_counter()
    py = integrate_python(0.0, 1.0, n)
    t_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    c = integrate_par.integrate(0.0, 1.0, n)
    t_c = time.perf_counter() - t0

    print(f"n = {n:,}")
    print(f"python : {py:.15f}  in {t_py:8.3f} s")
    print(f"C+OpenMP: {c:.15f}  in {t_c:8.3f} s   ({t_py / t_c:,.0f}x faster)")
    print(f"oracle : {expected:.15f}  (tol {tol})")
    ok_py = abs(py - expected) < tol
    ok_c = abs(c - expected) < tol
    print("agreement:", "python OK," if ok_py else "python FAIL,",
          "C OK" if ok_c else "C FAIL")
    raise SystemExit(0 if (ok_py and ok_c) else 1)


if __name__ == "__main__":
    main()

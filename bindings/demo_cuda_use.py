"""CUDA demonstration: the notebook's integral on the GPU, called from Python.

Run from bindings/ (on the GPU host) after building:  .venv/bin/python demo_cuda_use.py
"""
import json
import time

import integrate_par_cuda


def main():
    oracle = json.load(open("../problems/integrate/reference.json"))
    n = oracle["params"]["N_TEST"]
    expected = oracle["expected"]["EXPECTED"]
    tol = oracle["tol"]

    integrate_par_cuda.integrate(0.0, 1.0, 1000)   # GPU warm-up: context + JIT
    t0 = time.perf_counter()
    value = integrate_par_cuda.integrate(0.0, 1.0, n)
    elapsed = time.perf_counter() - t0

    ok = abs(value - expected) < tol
    print(f"GPU from Python: integral = {value:.15f} in {elapsed * 1000:.1f} ms")
    print(f"oracle: {expected:.15f} (tol {tol}) ->", "OK" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

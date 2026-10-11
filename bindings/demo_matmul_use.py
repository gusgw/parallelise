"""Array-signature demonstration: numpy matrices in, numpy matrix out, parallel C inside.

Run from bindings/ after building:  .venv/bin/python demo_matmul_use.py

Builds the notebook's matrices with numpy, multiplies them three ways -- the notebook's
pure-Python triple loop, numpy's optimised A @ B, and the wrapped parallel C kernel --
and checks all three against the notebook-derived oracle checksum.
"""
import json
import time

import numpy as np

import matmul_par


def main():
    oracle = json.load(open("../problems/matmul/reference.json"))
    n = oracle["params"]["N"]
    expected = oracle["expected"]["CHECKSUM"]
    tol = oracle["tol"]

    i = np.arange(n).reshape(n, 1)
    j = np.arange(n).reshape(1, n)
    A = (((i * 3 + j * 7) % 11) + 1).astype(np.float64)
    B = (((i * 5 + j * 2) % 13) + 1).astype(np.float64)

    def python_matmul(A, B, n):
        C = [[0.0] * n for _ in range(n)]
        for r in range(n):
            for c in range(n):
                s = 0.0
                for k in range(n):
                    s += A[r][k] * B[k][c]
                C[r][c] = s
        return C

    t0 = time.perf_counter()
    Cpy = python_matmul(A.tolist(), B.tolist(), n)
    t_py = time.perf_counter() - t0
    cs_py = sum(Cpy[r][c] for r in range(n) for c in range(n))

    t0 = time.perf_counter()
    Cnp = A @ B
    t_np = time.perf_counter() - t0

    t0 = time.perf_counter()
    Cc = matmul_par.matmul(A, B)
    t_c = time.perf_counter() - t0

    print(f"n = {n}")
    print(f"python loop : checksum {cs_py:.1f}  in {t_py:8.4f} s")
    print(f"numpy A @ B : checksum {Cnp.sum():.1f}  in {t_np:8.4f} s")
    print(f"C + OpenMP  : checksum {Cc.sum():.1f}  in {t_c:8.4f} s   ({t_py / t_c:,.0f}x over the loop)")
    print(f"oracle      : {expected:.1f}  (tol {tol})")
    ok = all(abs(v - expected) < tol for v in (cs_py, float(Cnp.sum()), float(Cc.sum())))
    print("agreement   :", "all OK" if ok else "MISMATCH")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

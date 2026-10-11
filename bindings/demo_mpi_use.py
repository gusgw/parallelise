"""The MPI end-goal demonstration: the notebook's algorithm, served by C+MPI from Python.

Launch with several processes -- the only difference from the OpenMP case:

    mpirun -np 4 .venv/bin/python demo_mpi_use.py

Importing mpi4py starts MPI; every process calls the wrapped collective kernel
together; each computes only its share and the combined result comes back to all of
them. Rank 0 checks it against the notebook-derived oracle.
"""
import json
import time

from mpi4py import MPI          # importing initialises MPI
import integrate_par_mpi

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

oracle = json.load(open("../problems/integrate/reference.json"))
n = oracle["params"]["N_TEST"]
expected = oracle["expected"]["EXPECTED"]
tol = oracle["tol"]

comm.Barrier()
t0 = time.perf_counter()
value = integrate_par_mpi.integrate(0.0, 1.0, n)
elapsed = time.perf_counter() - t0

if rank == 0:
    ok = abs(value - expected) < tol
    print(f"{size} processes: integral = {value:.15f} in {elapsed:.4f} s")
    print(f"oracle: {expected:.15f} (tol {tol})  ->", "OK" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)

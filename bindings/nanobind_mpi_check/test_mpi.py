"""Run under: mpirun -np K .venv/bin/python test_mpi.py

Importing mpi4py starts MPI. Each of the K processes calls the nanobind-wrapped C
function with mpi4py's communicator handle; the C code sums its own slice and combines
across processes with MPI_Allreduce. A correct pi at K>1 means the processes really did
share the work and message each other through the binding.
"""
import math
from mpi4py import MPI          # importing initialises MPI
import pi_mpi_ext

comm = MPI.COMM_WORLD
handle = comm.py2f()            # Fortran handle to hand to the C function
rank, size = pi_mpi_ext.rank_size(handle)

n = 100_000_000
value = pi_mpi_ext.pi_mpi(handle, n)

print(f"[process] I am rank {rank} of {size}", flush=True)
if rank == 0:
    err = abs(value - math.pi)
    ok = (size > 1) and (err < 1e-6)
    print(f"[rank 0] size={size}  pi={value:.12f}  error={err:.2e}")
    print("RESULT:", "PASS" if ok else "FAIL",
          "- multiple processes shared the work and combined to pi"
          if ok else "- expected size>1 and pi within 1e-6")

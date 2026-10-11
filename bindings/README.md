# Bindings: calling the finished kernels from Python

These are small wrappers, written with nanobind (a library that exposes C and C++
functions to Python), that let a researcher call a kernel written by an agent from
ordinary Python code. They answer the practical question behind the whole project: once
an agent has produced a parallel C kernel that passes the gate, how little must the
researcher change in the notebook to use it? In the examples the only change is one
import line, for instance `from integrate_par import integrate`.

The wrappers do not contain any kernels. You point the build at a work folder that holds
a passing kernel (an agent's output, or the known-good solution under
`problems/<name>/<backend>/reference/`), and CMake compiles that kernel together with the
wrapper into an importable Python module.

## Contents

| File | What it is |
| --- | --- |
| `CMakeLists.txt` | Builds whichever modules you ask for with `-DKERNEL_DIR`, `-DMPI_KERNEL_DIR`, `-DMATMUL_KERNEL_DIR`, `-DJACOBI_KERNEL_DIR` or `-DCUDA_KERNEL_DIR`. |
| `integrate_binding.cpp`, `integrate_mpi_binding.cpp`, `integrate_cuda_binding.cpp` | Scalar wrappers for the integral, for the OpenMP, MPI and CUDA kernels. They release the Python global interpreter lock around the kernel so that the kernel's own threads can use the cores. |
| `matmul_binding.cpp`, `jacobi_binding.cpp` | Array wrappers. NumPy arrays of 64-bit floats cross the boundary without copying on the way in; the result buffer is owned by a capsule so that Python frees it safely. |
| `demo_*.py` | One demonstration per wrapper. Each compares the notebook's own pure-Python loop with the wrapped kernel and checks both against the oracle in `problems/<name>/reference.json`. |
| `nanobind_mpi_check/` | A small, separate experiment that showed in the first week that MPI code can be called from Python: `mpi4py` hands its communicator to a C function, which sums its slice of a midpoint-rule calculation of pi and combines the slices with an all-reduce. |

## How they were built and tested in July 2026

Everything was built with CMake against nanobind 2.13, Python 3.14, Open MPI 5.0.10 and
GCC 16, and the demonstrations were run on a workstation. The CUDA wrapper was built
and run on a machine with an NVIDIA GPU. A typical build for the OpenMP integral is:

    cmake -S . -B build -DKERNEL_DIR=<folder holding integrate.c> \
          -Dnanobind_DIR=$(python -c 'import nanobind; print(nanobind.cmake_dir())')
    cmake --build build -j4
    cp build/integrate_par*.so .
    python demo_notebook_use.py

The MPI wrapper is run under `mpirun`, because the converted kernel is collective over
all processes: `mpirun -np 4 python demo_mpi_use.py`.

Measured results from those demonstrations, all matching the oracle within its
tolerance (these are single runs, not a benchmark study):

- Integral, OpenMP: the notebook's Python loop took 1.33 s and the wrapped kernel
  0.004 s, about 365 times faster.
- Integral, MPI: four processes were 3.4 times faster than one (0.0044 s against
  0.0152 s at ten million points). The one-process and four-process answers differed by
  6e-14, inside the rounding spread measured for the oracle.
- Integral, CUDA: 2.3 ms on the GPU.
- Matrix multiplication: the Python loop took 1.49 s, the wrapped kernel 0.012 s (about
  123 times faster) and NumPy's `A @ B` 0.004 s. The optimised library beats the simple
  kernel at this size; the point of the demonstration is the array-passing pattern.

## One trap worth knowing

The kernels are plain C functions with common names such as `integrate`. A Python
interpreter built with some tools exports a global function of that name, and a module's
internal call to its own function can then be captured by the interpreter's version
instead. The symptom was a silent `nan` from the CUDA module. The fix, already in
`CMakeLists.txt`, is to link every module with `-Wl,-Bsymbolic`, which makes each module
use its own definitions.

The compiled modules (`*.so`) and build folders are deliberately not kept in the
repository.

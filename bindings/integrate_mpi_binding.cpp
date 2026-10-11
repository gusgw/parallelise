// nanobind binding for the converted MPI integrate kernel. The kernel is collective
// over MPI_COMM_WORLD (every rank calls it together; each computes its share and the
// result is combined internally), so no communicator crosses the Python/C boundary.
// Launch:  mpirun -np K python script.py   -- importing mpi4py initialises MPI.
#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nb::literals;

extern "C" double integrate(double a, double b, long n);

NB_MODULE(integrate_par_mpi, m)
{
    m.def("integrate", &integrate, "a"_a, "b"_a, "n"_a,
          nb::call_guard<nb::gil_scoped_release>(),
          "Midpoint-rule integral of 4/(1+x^2) on [a, b] with n subintervals, "
          "computed collectively by the parallel (MPI) C kernel. Every rank must "
          "call it together; every rank returns the full result.");
}

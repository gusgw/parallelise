// nanobind binding for the converted integrate kernel: makes the parallel C callable
// from Python / a Jupyter notebook as  integrate_par.integrate(a, b, n).
// The kernel itself (integrate.c, OpenMP) is compiled alongside this file; the binding
// releases the GIL so the OpenMP threads own the cores while Python waits.
#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nb::literals;

extern "C" double integrate(double a, double b, long n);

NB_MODULE(integrate_par, m)
{
    m.def("integrate", &integrate, "a"_a, "b"_a, "n"_a,
          nb::call_guard<nb::gil_scoped_release>(),
          "Midpoint-rule integral of 4/(1+x^2) on [a, b] with n subintervals, "
          "computed by the parallel (OpenMP) C kernel.");
}

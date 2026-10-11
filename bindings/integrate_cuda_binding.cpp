// nanobind binding for the converted CUDA integrate kernel. Identical surface to the
// OpenMP wrapper -- the host function launches the GPU kernel and returns the finished
// integral; from Python the only difference is the module name.
#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nb::literals;

extern "C" double integrate(double a, double b, long n);

NB_MODULE(integrate_par_cuda, m)
{
    m.def("integrate", &integrate, "a"_a, "b"_a, "n"_a,
          nb::call_guard<nb::gil_scoped_release>(),
          "Midpoint-rule integral of 4/(1+x^2) on [a, b] with n subintervals, "
          "computed by the CUDA kernel on the GPU.");
}

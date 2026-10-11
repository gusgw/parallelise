// nanobind binding for the converted jacobi kernel: a numpy grid crosses the boundary.
// u_final = jacobi(u, iters) takes an n x n float64 C-contiguous array and returns a
// NEW array holding the grid after `iters` sweeps (the input is not modified). The
// kernel (jacobi.c, OpenMP) is compiled alongside; the GIL is released while it runs.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstring>
#include <stdexcept>

namespace nb = nanobind;
using namespace nb::literals;

extern "C" void jacobi(double *u, long n, int iters);

using Grid = nb::ndarray<double, nb::numpy, nb::ndim<2>, nb::c_contig>;

static Grid jacobi_py(Grid u, int iters)
{
    if (u.shape(0) != u.shape(1))
        throw std::invalid_argument("u must be square");
    if (iters < 0)
        throw std::invalid_argument("iters must be >= 0");
    const long n = (long)u.shape(0);

    double *out = new double[(size_t)n * n];
    std::memcpy(out, u.data(), sizeof(double) * (size_t)n * n);
    nb::capsule owner(out, [](void *p) noexcept { delete[] (double *)p; });

    {
        nb::gil_scoped_release release;
        jacobi(out, n, iters);
    }
    return Grid(out, {(size_t)n, (size_t)n}, owner);
}

NB_MODULE(jacobi_par, m)
{
    m.def("jacobi", &jacobi_py, "u"_a, "iters"_a,
          "Run `iters` Jacobi sweeps on the n x n grid with the parallel (OpenMP) C "
          "kernel; returns the final grid as a new array.");
}

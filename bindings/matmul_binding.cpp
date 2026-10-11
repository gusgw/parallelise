// nanobind binding for the converted matmul kernel: numpy arrays cross the boundary.
// C = matmul(A, B) takes two n x n float64 C-contiguous arrays and returns a new one.
// The kernel (matmul.c, OpenMP) is compiled alongside; the GIL is released so the
// OpenMP threads own the cores while Python waits.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <stdexcept>

namespace nb = nanobind;
using namespace nb::literals;

extern "C" void matmul(const double *A, const double *B, double *C, long n);

using Matrix = nb::ndarray<double, nb::numpy, nb::ndim<2>, nb::c_contig>;

static Matrix matmul_py(Matrix A, Matrix B)
{
    if (A.shape(0) != A.shape(1) || B.shape(0) != B.shape(1)
        || A.shape(0) != B.shape(0))
        throw std::invalid_argument("A and B must be square with the same n");
    const long n = (long)A.shape(0);

    // result buffer owned by a capsule so numpy frees it when done
    double *C = new double[(size_t)n * n];
    nb::capsule owner(C, [](void *p) noexcept { delete[] (double *)p; });

    {
        nb::gil_scoped_release release;
        matmul(A.data(), B.data(), C, n);
    }
    return Matrix(C, {(size_t)n, (size_t)n}, owner);
}

NB_MODULE(matmul_par, m)
{
    m.def("matmul", &matmul_py, "A"_a, "B"_a,
          "C = A @ B computed by the parallel (OpenMP) C kernel; "
          "takes and returns n x n float64 numpy arrays.");
}

#include <mpi.h>

#include "integrate.h"

/* Midpoint-rule estimate of the integral of 4/(1+x^2) over [a, b] with n
 * subintervals, distributed across MPI ranks: each rank sums a contiguous block
 * of the subintervals and the partial sums are combined with MPI_Allreduce, so
 * every rank returns the same total. The caller must have called MPI_Init. */
double integrate(double a, double b, long n)
{
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const double h = (b - a) / (double)n;

    /* block decomposition of [0, n) across the ranks (no gaps, no overlap) */
    const long lo = n * (long)rank / size;
    const long hi = n * (long)(rank + 1) / size;

    double local = 0.0;
    for (long i = lo; i < hi; i++) {
        const double x = a + ((double)i + 0.5) * h;
        local += 4.0 / (1.0 + x * x);
    }

    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global * h;
}

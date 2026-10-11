#include "integrate.h"
#include <mpi.h>
#include <math.h>

/*
 * Parallel midpoint-rule integration.
 * Each MPI rank computes a contiguous block of sub‑intervals and the partial
 * sums are reduced with MPI_Allreduce so that every rank receives the same
 * final integral value.
 */

double integrate(double a, double b, long n)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* width of each sub‑interval */
    double h = (b - a) / (double)n;

    /* block decomposition */
    long lo = (n * (long)rank) / (long)size;
    long hi = (n * (long)(rank + 1)) / (long)size;

    double local_sum = 0.0;
    for (long i = lo; i < hi; ++i) {
        double x = a + (i + 0.5) * h;
        local_sum += 4.0 / (1.0 + x * x);
    }

    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return global_sum * h;
}

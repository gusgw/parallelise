#include <mpi.h>
#include "integrate.h"

/* Parallel midpoint-rule integration of 4/(1+x^2) over [a,b].
 * Each rank sums its own block of subintervals and the partial sums are
 * combined with MPI_Allreduce to produce the same result on every rank.
 */

double integrate(double a, double b, long n)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (n <= 0) return 0.0; // degenerate case

    /* Determine the range of indices this rank will handle: [lo, hi) */
    long lo = (n * rank) / size;
    long hi = (n * (rank + 1)) / size;

    double h = (b - a) / (double)n;
    double local_sum = 0.0;
    for (long i = lo; i < hi; ++i) {
        double x = a + (i + 0.5) * h;
        local_sum += 4.0 / (1.0 + x * x);
    }

    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return global_sum * h;
}

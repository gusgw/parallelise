#include "kernel_a.h"

#include <mpi.h>

/* Numerical integration using midpoint rule to approximate pi.
 * Each rank computes a partial sum over its share of iterations,
 * then MPI_Allreduce combines the partial sums. */
double kernel_a(double a, double b, long n)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double h = (b - a) / n;

    /* Divide iterations among ranks: each rank gets a contiguous block */
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;

    double local_sum = 0.0;
    for (long i = lo; i < hi; i++) {
        double x = a + (i + 0.5) * h;
        local_sum += 4.0 / (1.0 + x * x);
    }

    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return global_sum * h;
}

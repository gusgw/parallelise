#include "kernel_a.h"
#include <mpi.h>

/*
 * Parallel implementation of the midpoint rule integration used in the
 * reference Python notebook. The computation is split evenly across MPI ranks.
 * Each rank computes a partial sum over its assigned interval and the partial
 * results are combined with an MPI_Allreduce so that every rank receives the
 * final integral value.
 */

double kernel_a(double a, double b, long n) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Determine the range of indices this rank will handle. */
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;

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

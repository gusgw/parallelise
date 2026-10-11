#include <mpi.h>
#include <stddef.h>

#include "integrate.h"

double integrate(double a, double b, long n)
{
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double h = (b - a) / n;

    /* Each rank computes its own block of the sum */
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;

    double local_sum = 0.0;
    for (long i = lo; i < hi; i++) {
        double x = a + (i + 0.5) * h;
        local_sum += 4.0 / (1.0 + x * x);
    }

    /* Combine partial sums from all ranks */
    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return global_sum * h;
}

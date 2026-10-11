#include <mpi.h>
#include "scan.h"

void scan(const double *in, double *out, long n)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long lo = (long)((double)n * rank / size);
    long hi = (long)((double)n * (rank + 1) / size);

    double local_sum = 0.0;
    for (long i = lo; i < hi; i++) {
        local_sum += in[i];
        out[i] = local_sum;
    }

    double offset = 0.0;
    /* MPI_Exscan computes the exclusive prefix sum of the local_sum values.
     * Rank r receives the sum of local_sum from ranks 0 to r-1.
     * Rank 0 receives MPI_PROC_NULL (or 0.0 depending on implementation). */
    MPI_Exscan(&local_sum, &offset, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0) {
        offset = 0.0;
    }

    for (long i = lo; i < hi; i++) {
        out[i] += offset;
    }
}

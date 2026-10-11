#include <mpi.h>
#include <stdlib.h>
#include "kernel_c.h"

void kernel_c(const double *in, double *out, long n)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    long local_n = hi - lo;

    // Allocate temporary buffer for local output
    double *local_out = NULL;
    if (local_n > 0) {
        local_out = (double *)malloc(local_n * sizeof(double));
    }

    double run = 0.0;
    for (long i = 0; i < local_n; ++i) {
        run += in[lo + i];
        local_out[i] = run;
    }
    double local_sum = run;
    double offset = 0.0;
    // Compute prefix sum of segment sums across ranks.
    MPI_Exscan(&local_sum, &offset, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    // Add offset to local output values
    for (long i = 0; i < local_n; ++i) {
        local_out[i] += offset;
    }
    // Copy local results into the appropriate slice of the global output array
    for (long i = 0; i < local_n; ++i) {
        out[lo + i] = local_out[i];
    }

    // Gather full output on rank 0 for verification or further use
    int *recvcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    for (int r = 0; r < size; ++r) {
        long r_lo = n * r / size;
        long r_hi = n * (r + 1) / size;
        recvcounts[r] = (int)(r_hi - r_lo);
        displs[r] = (int)r_lo;
    }
    MPI_Gatherv(local_out, (int)local_n, MPI_DOUBLE,
                out, recvcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    free(local_out);
    free(recvcounts);
    free(displs);
}

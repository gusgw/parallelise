#include <mpi.h>
#include <stdlib.h>

#include "kernel_c.h"

void kernel_c(const double *in, double *out, long n)
{
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Decompose work: each rank handles elements [lo, hi) */
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    long count = hi - lo;

    /* Local prefix sum on this rank's portion */
    double run = 0.0;
    for (long i = 0; i < count; i++) {
        run += in[lo + i];
        out[lo + i] = run;
    }

    /* Compute sum of this rank's block */
    double local_sum = run;
    double offset = 0.0;

    /* Exclusive scan: rank r gets sum of blocks 0..r-1 */
    MPI_Exscan(&local_sum, &offset, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) offset = 0.0;

    /* Add offset to local results */
    for (long i = 0; i < count; i++) {
        out[lo + i] += offset;
    }

    /* Gather results to rank 0 using Gatherv */
    /* Compute counts and displacements for all ranks */
    int *counts = malloc(sizeof(int) * size);
    int *displs = malloc(sizeof(int) * size);
    if (!counts || !displs) {
        free(counts);
        free(displs);
        return;
    }

    for (int r = 0; r < size; r++) {
        long r_lo = n * r / size;
        long r_hi = n * (r + 1) / size;
        counts[r] = (int)(r_hi - r_lo);
        displs[r] = (int)r_lo;
    }

    MPI_Gatherv(&out[lo], (int)count, MPI_DOUBLE,
                out, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    free(counts);
    free(displs);
}

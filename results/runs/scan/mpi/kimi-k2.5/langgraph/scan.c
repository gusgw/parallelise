#include "scan.h"
#include <mpi.h>

void scan(const double *in, double *out, long n)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Compute this rank's block: [lo, hi) */
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    long count = hi - lo;

    /* Step 1: Local inclusive scan of this rank's block */
    double local_sum = 0.0;
    for (long i = 0; i < count; i++) {
        local_sum += in[lo + i];
        out[lo + i] = local_sum;
    }

    /* Step 2: Get the starting offset from all previous ranks' block sums.
     * MPI_Exscan computes the exclusive prefix sum: rank r receives sum of
     * block totals from ranks 0..r-1. Rank 0 receives 0. */
    double offset = 0.0;
    MPI_Exscan(&local_sum, &offset, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    /* Step 3: Add offset to each element in the local block */
    for (long i = 0; i < count; i++) {
        out[lo + i] += offset;
    }
}

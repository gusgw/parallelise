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

    /* Step 1: Local inclusive scan of the block */
    double local_sum = 0.0;
    for (long i = 0; i < count; i++) {
        local_sum += in[lo + i];
        out[lo + i] = local_sum;
    }

    /* Step 2: Get the offset from all previous ranks using MPI_Exscan */
    double offset = 0.0;
    MPI_Exscan(&local_sum, &offset, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    /* Note: rank 0 receives MPI_PROC_NULL and offset remains 0.0 */

    /* Step 3: Add offset to each element in the local block */
    for (long i = 0; i < count; i++) {
        out[lo + i] += offset;
    }
}

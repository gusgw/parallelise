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

    /* Local inclusive scan of the block */
    double local_sum = 0.0;
    for (long i = 0; i < count; i++) {
        local_sum += in[lo + i];
        out[i] = local_sum;
    }

    /* Get the starting offset from all previous ranks using exclusive scan */
    double offset = 0.0;
    MPI_Exscan(&local_sum, &offset, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    /* Note: rank 0 gets MPI_PROC_NULL result which is 0 (uninitialized but MPI guarantees 0) */
    /* Actually MPI_Exscan on rank 0 leaves recvbuf unchanged, but we want 0 */
    if (rank == 0) {
        offset = 0.0;
    }

    /* Add offset to each element in the local block */
    for (long i = 0; i < count; i++) {
        out[i] += offset;
    }
}

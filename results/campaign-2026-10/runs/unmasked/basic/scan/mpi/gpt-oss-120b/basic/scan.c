#include <mpi.h>
#include <stddef.h>

/*
 * Parallel inclusive prefix sum (scan).
 * Each rank holds a full copy of the input array `in` of length n.
 * The output array `out` is distributed: rank r must contain the correct
 * values for the contiguous block [lo, hi) where
 *   lo = n * rank / size
 *   hi = n * (rank + 1) / size
 *
 * The implementation performs a local scan of the owned block, then uses an
 * exclusive prefix sum (MPI_Exscan) of the block totals to compute the offset
 * that each rank must add to its local results.
 */

void scan(const double *in, double *out, long n)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Determine the range owned by this rank */
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;

    /* Local inclusive scan */
    double run = 0.0;
    for (long i = lo; i < hi; ++i) {
        run += in[i];
        out[i] = run;
    }

    /* Total of this block (last element of the local scan) */
    double local_total = 0.0;
    if (hi > lo) {
        local_total = out[hi - 1];
    }

    /* Exclusive prefix sum of block totals across ranks */
    double offset = 0.0;
    MPI_Exscan(&local_total, &offset, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        offset = 0.0; /* MPI_Exscan leaves recvbuf undefined for rank 0 */
    }

    /* Add the offset to the locally computed results */
    for (long i = lo; i < hi; ++i) {
        out[i] += offset;
    }
}

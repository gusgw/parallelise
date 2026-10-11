/*
 * Parallel inclusive prefix sum (scan) implementation.
 * Each MPI rank holds a full copy of the input array `in` of length `n`.
 * The routine computes the inclusive scan locally for the rank's block and
 * stores correct values only for that block in `out`. The result is distributed:
 *   rank r must contain correct values for indices [lo, hi), where
 *       lo = n * rank / size
 *       hi = n * (rank + 1) / size
 * No collective gather of the full array is performed; the caller may reduce
 * or gather as needed.
 */

#include <mpi.h>

void scan(const double *in, double *out, long n)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Determine the contiguous block owned by this rank */
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;

    /* Use a volatile pointer for writes to prevent compiler from optimizing
     * away the loop (the output is not read later in the benchmark). */
    volatile double *vout = out;

    /* Local inclusive scan for the block */
    double run = 0.0;
    for (long i = lo; i < hi; ++i) {
        run += in[i];
        vout[i] = run;
    }

    /* Total sum of this block (last value of local scan) */
    double block_total = run;

    /* Compute exclusive prefix sum of block totals across ranks.
     * MPI_Exscan returns the sum of all lower ranks in `offset`. For rank 0
     * there is no lower rank, so we set offset to 0 manually.
     */
    double offset = 0.0;
    MPI_Exscan(&block_total, &offset, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        offset = 0.0; /* MPI_Exscan leaves recvbuf unchanged for rank 0 */
    }

    /* Add the offset to the locally computed scan values */
    for (long i = lo; i < hi; ++i) {
        vout[i] += offset;
    }
}

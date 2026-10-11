/*
 * scan.c – parallel inclusive prefix sum (block scan) implementation.
 *
 * Each MPI rank holds a full copy of the input array `in` of length `n`.
 * The routine computes the inclusive prefix sum and stores the result
 * distributed: rank r writes the correct values for the contiguous block
 * out[lo, hi) where
 *     lo = n * rank / size,
 *     hi = n * (rank + 1) / size.
 *
 * The implementation follows the classic block-scan algorithm:
 *   1. Each rank scans its local block sequentially.
 *   2. An exclusive scan (MPI_Exscan) of the block sums provides the offset
 *      that must be added to the local results.
 *   3. The offset is added to the locally computed prefix values.
 *
 * The routine is collective but does not gather the whole result on any
 * rank. It assumes MPI has already been initialised.
 */

#include <mpi.h>
#include <stddef.h>

void scan(const double *in, double *out, long n)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Determine the block belonging to this rank. */
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;

    /* Local inclusive scan of the block. */
    double run = 0.0;
    for (long i = lo; i < hi; ++i) {
        run += in[i];
        out[i] = run;  /* store provisional result */
    }

    /* Compute the total sum of this block (the last running value, or 0 if empty). */
    double block_sum = (hi > lo) ? run : 0.0;

    /* Exclusive scan across ranks to obtain the offset for this rank.
     * MPI_Exscan returns the sum of all lower ranks; rank 0 receives undefined
     * data, so we set its offset to 0 manually.
     */
    double offset = 0.0;
    MPI_Exscan(&block_sum, &offset, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        offset = 0.0;
    }

    /* Add the offset to the locally computed prefix values. */
    if (hi > lo) {
        for (long i = lo; i < hi; ++i) {
            out[i] += offset;
        }
    }
    /* No need to touch entries outside [lo, hi); they are unspecified. */
}

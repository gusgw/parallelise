/* histogram.c — parallel MPI implementation of histogram.h
 *
 * Strategy (as specified in histogram.h):
 *   1. Each rank bins only its block [lo, hi) of the n input values into a
 *      local partial counts array (zeroed first).
 *   2. MPI_Allreduce with MPI_SUM combines the per-rank partial counts into
 *      the full histogram on every rank.
 *
 * The kernel is COLLECTIVE: MPI is already initialised by the caller; we do
 * NOT call MPI_Init or MPI_Finalize here.
 */

#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "histogram.h"

void histogram(const double *x, long n, long *counts, int nbins)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Block decomposition: rank r owns indices [lo, hi) */
    long lo = (long)nbins >= 0 ? n * rank / size : 0; /* silence analyser */
    lo = n * rank / size;
    long hi = n * (rank + 1) / size;

    /* Zero the output array (will accumulate the Allreduce result) */
    memset(counts, 0, sizeof(long) * (size_t)nbins);

    /* Bin this rank's slice into counts directly (it is already zeroed) */
    for (long i = lo; i < hi; i++) {
        int b = (int)(x[i] * nbins);
        if (b >= nbins) b = nbins - 1;
        counts[b]++;
    }

    /* Element-wise sum reduction: every rank gets the full histogram */
    MPI_Allreduce(MPI_IN_PLACE, counts, nbins, MPI_LONG, MPI_SUM,
                  MPI_COMM_WORLD);
}

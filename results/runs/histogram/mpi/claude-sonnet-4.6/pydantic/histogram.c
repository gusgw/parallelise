/* histogram.c -- parallel MPI histogram kernel implementing histogram.h
 *
 * Strategy (as specified in histogram.h):
 *   1. Each rank bins only its block [lo, hi) of the n input values into a
 *      local counts array (initialised to zero).
 *   2. An element-wise MPI_Allreduce (MPI_SUM) over nbins MPI_LONG values
 *      combines every rank's partial counts into the full histogram on every rank.
 */

#include <stdlib.h>   /* malloc, free */
#include <string.h>   /* memset */
#include <mpi.h>

#include "histogram.h"

void histogram(const double *x, long n, long *counts, int nbins)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Zero the output array (caller-supplied; we own its contents). */
    memset(counts, 0, sizeof(long) * (size_t)nbins);

    /* Decompose value indices across ranks: rank r owns [lo, hi). */
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;

    /* Accumulate partial histogram for this rank's slice. */
    for (long i = lo; i < hi; i++) {
        int b = (int)(x[i] * nbins);
        if (b >= nbins) b = nbins - 1;
        counts[b]++;
    }

    /* Reduce all partial histograms into the full histogram on every rank. */
    MPI_Allreduce(MPI_IN_PLACE, counts, nbins, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
}

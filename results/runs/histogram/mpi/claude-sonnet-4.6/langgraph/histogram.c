/* histogram.c – parallel MPI histogram kernel
 *
 * Each rank bins only its own block [lo, hi) of the n input values into a
 * local counts array, then MPI_Allreduce (MPI_SUM) combines all per-rank
 * partial counts into the full histogram on every rank.
 */

#include <stdlib.h>   /* malloc, free */
#include <string.h>   /* memset       */
#include <mpi.h>

#include "histogram.h"

void histogram(const double *x, long n, long *counts, int nbins)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Block-decompose the n value indices across ranks */
    long lo = (long)rank       * n / size;
    long hi = (long)(rank + 1) * n / size;

    /* Zero the caller-supplied output array (will hold the global result) */
    memset(counts, 0, sizeof(long) * nbins);

    /* Accumulate partial counts for this rank's slice */
    long *local = (long *)malloc(sizeof(long) * nbins);
    if (!local) {
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }
    memset(local, 0, sizeof(long) * nbins);

    for (long i = lo; i < hi; i++) {
        int b = (int)(x[i] * nbins);
        if (b >= nbins) b = nbins - 1;
        local[b]++;
    }

    /* Sum partial histograms across all ranks; every rank receives the result */
    MPI_Allreduce(local, counts, nbins, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

    free(local);
}

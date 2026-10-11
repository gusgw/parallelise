#include "histogram.h"

#include <mpi.h>
#include <stdlib.h>
#include <string.h>

/* Histogram of n values x[0..n-1], all in [0, 1), into nbins equal-width bins.
 * Each rank computes a partial histogram for its block of values, then all
 * ranks combine their partial results using MPI_Allreduce with MPI_SUM. */
void histogram(const double *x, long n, long *counts, int nbins)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Zero the output counts */
    for (int b = 0; b < nbins; b++) {
        counts[b] = 0;
    }

    /* Compute local block bounds [lo, hi) */
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;

    /* Local histogram for this rank's block */
    for (long i = lo; i < hi; i++) {
        int b = (int)(x[i] * nbins);
        if (b >= nbins) {
            b = nbins - 1;
        }
        counts[b]++;
    }

    /* Combine all partial histograms using element-wise sum reduction */
    if (size > 1) {
        long *local_counts = malloc(sizeof(long) * nbins);
        if (!local_counts) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        memcpy(local_counts, counts, sizeof(long) * nbins);
        MPI_Allreduce(local_counts, counts, nbins, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
        free(local_counts);
    }
}

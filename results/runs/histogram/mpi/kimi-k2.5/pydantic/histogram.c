#include "histogram.h"
#include <mpi.h>

void histogram(const double *x, long n, long *counts, int nbins)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Decompose the work: each rank handles indices [lo, hi) */
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;

    /* Zero local counts */
    long *local_counts = counts;  /* Use the output array as local storage */
    for (int b = 0; b < nbins; b++) {
        local_counts[b] = 0;
    }

    /* Bin the local portion of values */
    for (long i = lo; i < hi; i++) {
        double v = x[i];
        int b = (int)(v * nbins);
        if (b >= nbins) {
            b = nbins - 1;
        }
        local_counts[b]++;
    }

    /* Combine all local counts into global counts on every rank */
    /* Use MPI_Allreduce with MPI_SUM to sum up all per-rank histograms */
    if (size > 1) {
        MPI_Allreduce(MPI_IN_PLACE, counts, nbins, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    }
}

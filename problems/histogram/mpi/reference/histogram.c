#include <mpi.h>

#include "histogram.h"

/* Reference solution (used only to validate the harness -- NOT shown to the agents).
 * Block decomposition of the value indices: each rank bins its block [lo, hi) into its
 * own counts array, then MPI_Allreduce sums the per-rank counts element-wise so every
 * rank ends with the full histogram. Counts are exact integers, so the combination
 * order cannot change the result. Collective; assumes MPI already initialised. */
void histogram(const double *x, long n, long *counts, int nbins)
{
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (int b = 0; b < nbins; b++) counts[b] = 0;

    const long lo = n * rank / size;
    const long hi = n * (rank + 1) / size;
    for (long i = lo; i < hi; i++) {
        int b = (int)(x[i] * (double)nbins);
        if (b >= nbins) b = nbins - 1;
        counts[b]++;
    }

    MPI_Allreduce(MPI_IN_PLACE, counts, nbins, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
}

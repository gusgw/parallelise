#include "histogram.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

void histogram(const double *x, long n, long *counts, int nbins)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Zero the output array on all ranks */
    for (int b = 0; b < nbins; b++) {
        counts[b] = 0;
    }

    /* Determine local range: each rank processes indices [lo, hi) */
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;

    /* Local histogram computation */
    for (long i = lo; i < hi; i++) {
        int b = (int)(x[i] * nbins);
        if (b >= nbins) {
            b = nbins - 1;
        }
        counts[b]++;
    }

    /* Combine per-rank counts using element-wise sum reduction */
    /* Each rank contributes its local counts, result is the full histogram on all ranks */
    if (size > 1) {
        long *local_counts = (long *)malloc(nbins * sizeof(long));
        if (local_counts == NULL) {
            /* In case of allocation failure, we still need to participate in collective */
            /* Use counts as-is (already has local values) */
            MPI_Allreduce(MPI_IN_PLACE, counts, nbins, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
        } else {
            memcpy(local_counts, counts, nbins * sizeof(long));
            MPI_Allreduce(local_counts, counts, nbins, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
            free(local_counts);
        }
    }
}

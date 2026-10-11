#include <mpi.h>
#include <string.h>
#include <stdlib.h>
#include "histogram.h"

void histogram(const double *x, long n, long *counts, int nbins)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;

    long *local_counts = (long *)calloc(nbins, sizeof(long));

    for (long i = lo; i < hi; i++) {
        int b = (int)(x[i] * nbins);
        if (b >= nbins) {
            b = nbins - 1;
        }
        if (b < 0) {
            b = 0;
        }
        local_counts[b]++;
    }

    MPI_Allreduce(local_counts, counts, nbins, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

    free(local_counts);
}

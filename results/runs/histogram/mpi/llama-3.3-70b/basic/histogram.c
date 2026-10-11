#include "histogram.h"
#include "problem.h"
#include <mpi.h>
#include <stdlib.h>

void histogram(const double *x, long n, long *counts, int nbins)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long *local_counts = (long *)malloc(nbins * sizeof(long));
    for (int b = 0; b < nbins; b++) local_counts[b] = 0;

    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    for (long i = lo; i < hi; i++) {
        int b = (int)(x[i] * nbins);
        if (b >= nbins) b = nbins - 1;
        local_counts[b] += 1;
    }

    MPI_Allreduce(local_counts, counts, nbins, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    free(local_counts);
}

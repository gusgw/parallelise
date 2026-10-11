#include "kernel_d.h"
#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void cgsolve(double *x, const double *b, long m, int iters) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long lo = m * rank / size;
    long hi = m * (rank + 1) / size;

    double *x_local = malloc((hi - lo) * sizeof(double));
    double *b_local = malloc((hi - lo) * sizeof(double));

    int *sendcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));

    for (int i = 0; i < size; i++) {
        sendcounts[i] = m / size;
        displs[i] = i * (m / size);
    }

    MPI_Scatterv(b, sendcounts, displs, MPI_DOUBLE, b_local, hi - lo, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < iters; i++) {
        // Perform CG iteration
        for (long j = lo; j < hi; j++) {
            x_local[j - lo] = b_local[j - lo];
        }
    }

    MPI_Gatherv(x_local, hi - lo, MPI_DOUBLE, x, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    free(x_local);
    free(b_local);
    free(sendcounts);
    free(displs);
}

#include "kernel_d.h"
#include "reference.h"
#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void cgsolve(double *x, const double *b, long m, int iters) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    long lo = m * rank / size;
    long hi = m * (rank + 1) / size;
    double *x_local = malloc((hi - lo) * sizeof(double));
    double *b_local = malloc((hi - lo) * sizeof(double));
    int sendcounts[size];
    int displs[size];
    for (int i = 0; i < size; i++) {
        sendcounts[i] = (hi - lo);
        displs[i] = i * (hi - lo);
    }
    MPI_Scatterv(b, sendcounts, displs, MPI_DOUBLE, b_local, hi - lo, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (int i = 0; i < iters; i++) {
        // Perform CG iteration
        for (long j = lo; j < hi; j++) {
            x_local[j - lo] = b_local[j - lo];
        }
    }
    int recvcounts[size];
    int recvdispls[size];
    for (int i = 0; i < size; i++) {
        recvcounts[i] = (hi - lo);
        recvdispls[i] = i * (hi - lo);
    }
    MPI_Gatherv(x_local, hi - lo, MPI_DOUBLE, x, recvcounts, recvdispls, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    free(x_local);
    free(b_local);
}

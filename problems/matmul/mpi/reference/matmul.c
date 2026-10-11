#include <mpi.h>
#include <stdlib.h>

#include "matmul.h"

/* Reference solution (used only to validate the harness -- NOT shown to the agents).
 * Row-block decomposition: each rank computes rows [lo, hi) of C, then MPI_Allgatherv
 * assembles the full C on every rank. Collective; assumes MPI already initialised. */
void matmul(const double *A, const double *B, double *C, long n)
{
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const long lo = n * rank / size;
    const long hi = n * (rank + 1) / size;
    for (long i = lo; i < hi; i++)
        for (long j = 0; j < n; j++) {
            double s = 0.0;
            for (long k = 0; k < n; k++) s += A[i * n + k] * B[k * n + j];
            C[i * n + j] = s;
        }

    int *counts = malloc(sizeof(int) * size);
    int *displs = malloc(sizeof(int) * size);
    for (int r = 0; r < size; r++) {
        long rlo = n * r / size, rhi = n * (r + 1) / size;
        counts[r] = (int)((rhi - rlo) * n);
        displs[r] = (int)(rlo * n);
    }
    /* each rank's block is already at displs[rank] in C -> gather in place */
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   C, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    free(counts);
    free(displs);
}

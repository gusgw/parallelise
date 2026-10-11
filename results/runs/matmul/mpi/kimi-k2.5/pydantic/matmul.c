#include "matmul.h"
#include <mpi.h>
#include <stdlib.h>

void matmul(const double *A, const double *B, double *C, long n)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Decompose rows of C across ranks: each rank computes rows [lo, hi) */
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    long local_rows = hi - lo;

    /* Allocate buffer for local result */
    double *local_C = (double *)malloc(local_rows * n * sizeof(double));
    if (!local_C) return;

    /* Compute local block of C */
    for (long i = 0; i < local_rows; i++) {
        long global_i = lo + i;
        for (long j = 0; j < n; j++) {
            double s = 0.0;
            for (long k = 0; k < n; k++) {
                s += A[global_i * n + k] * B[k * n + j];
            }
            local_C[i * n + j] = s;
        }
    }

    /* Gather counts from all ranks - needed by all ranks for Allgatherv */
    int *recvcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    if (!recvcounts || !displs) {
        free(local_C);
        free(recvcounts);
        free(displs);
        return;
    }

    int local_count = (int)(local_rows * n);
    MPI_Allgather(&local_count, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);

    /* Compute displacements */
    displs[0] = 0;
    for (int i = 1; i < size; i++) {
        displs[i] = displs[i - 1] + recvcounts[i - 1];
    }

    /* Allgatherv to distribute results to all ranks */
    MPI_Allgatherv(local_C, local_count, MPI_DOUBLE,
                   C, recvcounts, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    free(local_C);
    free(recvcounts);
    free(displs);
}

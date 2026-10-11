#include "matmul.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

void matmul(const double *A, const double *B, double *C, long n)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Compute row block for this rank: [row_start, row_end) */
    long row_start = n * rank / size;
    long row_end = n * (rank + 1) / size;
    long local_rows = row_end - row_start;

    /* Allocate temporary buffer for local result */
    double *local_C = (double *)malloc(local_rows * n * sizeof(double));
    if (!local_C) {
        /* In case of allocation failure, we still need to participate in collective */
        local_C = NULL;
    }

    /* Compute local block of C: C[row_start:row_end, :] = A[row_start:row_end, :] * B */
    if (local_C) {
        for (long i = 0; i < local_rows; i++) {
            for (long j = 0; j < n; j++) {
                double s = 0.0;
                long global_i = row_start + i;
                for (long k = 0; k < n; k++) {
                    s += A[global_i * n + k] * B[k * n + j];
                }
                local_C[i * n + j] = s;
            }
        }
    }

    /* Gather local row counts from all ranks - needed on ALL ranks for Allgatherv */
    int *recvcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));

    /* Each rank sends its local row count */
    int local_count = (int)(local_rows * n);
    MPI_Allgather(&local_count, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);

    /* Compute displacements on all ranks */
    displs[0] = 0;
    for (int i = 1; i < size; i++) {
        displs[i] = displs[i - 1] + recvcounts[i - 1];
    }

    /* Allgatherv to assemble full C on all ranks */
    MPI_Allgatherv(local_C, local_count, MPI_DOUBLE,
                   C, recvcounts, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    /* Clean up */
    free(local_C);
    free(recvcounts);
    free(displs);
}

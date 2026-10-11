#include "matmul.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

void matmul(const double *A, const double *B, double *C, long n)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Compute row block for this rank */
    long row_lo = n * rank / size;
    long row_hi = n * (rank + 1) / size;
    long nrows = row_hi - row_lo;

    /* Allocate temporary buffer for local rows of C */
    double *C_local = (double *)malloc(nrows * n * sizeof(double));
    if (C_local == NULL) {
        /* In case of allocation failure, we still need to participate in collective */
        /* Set to zero and continue to avoid deadlock */
        C_local = NULL;
    }

    if (C_local != NULL) {
        /* Compute local rows of C: C[i][j] = sum_k A[i][k] * B[k][j] */
        for (long i = 0; i < nrows; i++) {
            long global_i = row_lo + i;
            for (long j = 0; j < n; j++) {
                double s = 0.0;
                for (long k = 0; k < n; k++) {
                    s += A[global_i * n + k] * B[k * n + j];
                }
                C_local[i * n + j] = s;
            }
        }
    }

    /* Gather all row blocks to all ranks using MPI_Allgatherv */
    /* First, gather the counts from each rank */
    int *recvcounts = NULL;
    int *displs = NULL;
    
    if (rank == 0) {
        recvcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
    }

    int local_count = (int)(nrows * n);
    MPI_Gather(&local_count, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; i++) {
            displs[i] = displs[i - 1] + recvcounts[i - 1];
        }
    }

    /* Broadcast recvcounts and displs to all ranks */
    if (rank != 0) {
        recvcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
    }
    MPI_Bcast(recvcounts, size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displs, size, MPI_INT, 0, MPI_COMM_WORLD);

    /* Allgatherv to combine all local results */
    MPI_Allgatherv(C_local, local_count, MPI_DOUBLE,
                   C, recvcounts, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    free(C_local);
    free(recvcounts);
    free(displs);
}

/* matmul.c -- parallel dense matrix multiply C = A * B
 *
 * Decomposition: rows of C are split across MPI ranks.
 *   rank r owns rows  lo = n*r/size  ..  hi = n*(r+1)/size  (exclusive)
 * Each rank computes its block of rows independently (no communication during
 * the multiply), then MPI_Allgatherv assembles the full C on every rank.
 *
 * Collective: MPI must already be initialised by the caller.
 * Do NOT call MPI_Init / MPI_Finalize here.
 */

#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#include "matmul.h"

void matmul(const double *A, const double *B, double *C, long n)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Row range owned by this rank */
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    long local_rows = hi - lo;

    /* Allocate a local buffer for this rank's block of C */
    double *local_C = (double *)malloc(sizeof(double) * local_rows * n);
    if (!local_C) {
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }

    /* Compute C[lo..hi-1][:] = A[lo..hi-1][:] * B */
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

    /* Build recvcounts and displacements for MPI_Allgatherv.
     * Rank r contributes (n*(r+1)/size - n*r/size) * n doubles,
     * starting at offset (n*r/size) * n doubles into C. */
    int *recvcounts = (int *)malloc(sizeof(int) * size);
    int *displs     = (int *)malloc(sizeof(int) * size);
    if (!recvcounts || !displs) {
        free(local_C);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }

    for (int r = 0; r < size; r++) {
        long r_lo = n * r / size;
        long r_hi = n * (r + 1) / size;
        recvcounts[r] = (int)((r_hi - r_lo) * n);
        displs[r]     = (int)(r_lo * n);
    }

    /* Gather all blocks into C on every rank */
    MPI_Allgatherv(local_C, (int)(local_rows * n), MPI_DOUBLE,
                   C, recvcounts, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    free(local_C);
    free(recvcounts);
    free(displs);
}

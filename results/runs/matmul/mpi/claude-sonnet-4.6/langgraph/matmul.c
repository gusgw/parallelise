/* matmul.c -- parallel dense matrix multiply C = A * B
 *
 * Strategy: decompose the rows of C across MPI ranks.
 *   rank r owns rows  lo = n*r/size  ..  hi = n*(r+1)/size  (exclusive).
 * Each rank computes its local block of C independently (no communication
 * needed for the multiply itself because every rank holds full copies of A
 * and B).  The full C is then assembled on every rank with MPI_Allgatherv.
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

    /* Allocate a local buffer for this rank's block of C rows */
    double *local_C = (double *)malloc(sizeof(double) * local_rows * n);
    if (!local_C) {
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }

    /* Compute local block: C[i][j] = sum_k A[i][k] * B[k][j]
     * i is a global row index in [lo, hi). */
    for (long i = lo; i < hi; i++) {
        for (long j = 0; j < n; j++) {
            double s = 0.0;
            for (long k = 0; k < n; k++) {
                s += A[i * n + k] * B[k * n + j];
            }
            local_C[(i - lo) * n + j] = s;
        }
    }

    /* Build the displacement and count arrays for MPI_Allgatherv */
    int *recvcounts = (int *)malloc(sizeof(int) * (size_t)size);
    int *displs     = (int *)malloc(sizeof(int) * (size_t)size);
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

    /* Gather all row-blocks into C on every rank */
    MPI_Allgatherv(local_C, (int)(local_rows * n), MPI_DOUBLE,
                   C, recvcounts, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    free(local_C);
    free(recvcounts);
    free(displs);
}

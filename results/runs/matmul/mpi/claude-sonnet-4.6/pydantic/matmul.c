/* matmul.c — parallel dense matrix multiply C = A * B (row-block decomposition).
 *
 * Strategy:
 *   - Each rank computes a contiguous block of rows of C.
 *     lo = n * rank / size,  hi = n * (rank+1) / size   (integer division)
 *   - Each rank owns (hi - lo) rows of C, each of length n.
 *   - After local computation, MPI_Allgatherv assembles the full C on every rank.
 *
 * MPI is already initialised by the caller; we do NOT call MPI_Init/MPI_Finalize.
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
    long local_rows = hi - lo;          /* may be 0 for excess ranks */

    /* Compute the local block of C: rows [lo, hi) */
    /* We write directly into the correct slice of C to avoid an extra copy. */
    for (long i = lo; i < hi; i++) {
        for (long j = 0; j < n; j++) {
            double s = 0.0;
            for (long k = 0; k < n; k++) {
                s += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = s;
        }
    }

    /* Build the displacement / count arrays for MPI_Allgatherv */
    int *recvcounts = (int *)malloc(sizeof(int) * (size_t)size);
    int *displs     = (int *)malloc(sizeof(int) * (size_t)size);
    if (!recvcounts || !displs) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int r = 0; r < size; r++) {
        long r_lo   = n * r / size;
        long r_hi   = n * (r + 1) / size;
        recvcounts[r] = (int)((r_hi - r_lo) * n);
        displs[r]     = (int)(r_lo * n);
    }

    /* Each rank sends its local slice; every rank receives the full matrix */
    MPI_Allgatherv(
        C + lo * n,          /* sendbuf  — local slice already written above */
        (int)(local_rows * n),
        MPI_DOUBLE,
        C,                   /* recvbuf  — full output matrix                */
        recvcounts,
        displs,
        MPI_DOUBLE,
        MPI_COMM_WORLD
    );

    free(recvcounts);
    free(displs);
}

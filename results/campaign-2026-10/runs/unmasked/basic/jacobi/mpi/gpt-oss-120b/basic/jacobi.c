#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "jacobi.h"

/* Parallel Jacobi stencil.
 * Each MPI rank owns a contiguous block of rows of the n x n grid.
 * The grid is double-buffered: cur holds the current values, next holds the updated values.
 * Halo rows are exchanged each sweep via MPI_Sendrecv.
 * After all iterations, the full grid is assembled on every rank using MPI_Allgatherv.
 */

void jacobi(double *u, long n, int iters)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Determine the range of rows owned by this rank: [lo, hi) */
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    long local_rows = hi - lo;               /* number of owned rows */

    /* Allocate double buffers with two halo rows (top and bottom). */
    double *cur = (double *)malloc((local_rows + 2) * n * sizeof(double));
    double *next = (double *)malloc((local_rows + 2) * n * sizeof(double));
    if (!cur || !next) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Initialise cur with the portion of the global grid supplied in u.
     * The ghost rows are left undefined for now; they will be filled by halo exchange.
     */
    for (long i = 0; i < local_rows; ++i) {
        long global_i = lo + i;
        memcpy(&cur[(i + 1) * n], &u[global_i * n], n * sizeof(double));
    }
    /* Zero the ghost rows (optional, not strictly needed). */
    memset(cur, 0, n * sizeof(double));
    memset(&cur[(local_rows + 1) * n], 0, n * sizeof(double));

    /* Main iteration loop */
    for (int it = 0; it < iters; ++it) {
        /* Exchange halo rows with neighbours, using MPI_PROC_NULL for missing neighbours */
        int src = (rank == 0) ? MPI_PROC_NULL : rank - 1;
        int dst = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;
        /* Send our first owned row upward, receive bottom halo */
        MPI_Sendrecv(&cur[1 * n], n, MPI_DOUBLE, src, 0,
                     &cur[(local_rows + 1) * n], n, MPI_DOUBLE, dst, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        /* Send our last owned row downward, receive top halo */
        MPI_Sendrecv(&cur[local_rows * n], n, MPI_DOUBLE, dst, 1,
                     &cur[0], n, MPI_DOUBLE, src, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Update interior rows */
        for (long i = 1; i <= local_rows; ++i) {
            long global_i = lo + (i - 1);
            /* Fixed boundary rows (top row i==0 or bottom row i==n-1) are not updated */
            if (global_i == 0 || global_i == n - 1) {
                memcpy(&next[i * n], &cur[i * n], n * sizeof(double));
                continue;
            }
            /* Keep left and right edges unchanged */
            next[i * n + 0] = cur[i * n + 0];
            next[i * n + n - 1] = cur[i * n + n - 1];
            for (long j = 1; j < n - 1; ++j) {
                next[i * n + j] = 0.25 * (cur[(i - 1) * n + j] +
                                          cur[(i + 1) * n + j] +
                                          cur[i * n + (j - 1)] +
                                          cur[i * n + (j + 1)]);
            }
        }
        /* Swap buffers */
        double *tmp = cur;
        cur = next;
        next = tmp;
    }

    /* Gather the full grid onto every rank using Allgatherv */
    int *recvcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    for (int r = 0; r < size; ++r) {
        long r_lo = n * r / size;
        long r_hi = n * (r + 1) / size;
        recvcounts[r] = (int)((r_hi - r_lo) * n);
        displs[r] = (int)(r_lo * n);
    }
    MPI_Allgatherv(&cur[1 * n], (int)(local_rows * n), MPI_DOUBLE,
                   u, recvcounts, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    free(recvcounts);
    free(displs);
    free(cur);
    free(next);
}

#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include "jacobi.h"
#include "problem.h"

/* Parallel Jacobi 5‑point stencil.
 * Input grid `u` is n×n row‑major with fixed boundaries (top row = 1.0, others = 0.0).
 * Performs `iters` sweeps using double buffering and halo exchange.
 * On return every rank holds the full final grid in `u`.
 */
void jacobi(double *u, long n, int iters)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Row decomposition: each rank owns rows [lo, hi)
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    long local_rows = hi - lo;               // number of rows owned by this rank

    // Allocate two buffers with two halo rows (one above, one below)
    double *buf0 = malloc(sizeof(double) * (local_rows + 2) * n);
    double *buf1 = malloc(sizeof(double) * (local_rows + 2) * n);
    if (!buf0 || !buf1) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Copy owned rows into interior of buf0 (halo rows left untouched for now)
    for (long i = 0; i < local_rows; ++i) {
        memcpy(&buf0[(i + 1) * n], &u[(lo + i) * n], n * sizeof(double));
    }
    // Initialise halo rows that correspond to global fixed boundaries.
    if (rank == 0) {
        // top boundary row (i==0) is hot
        memcpy(&buf0[0 * n], &u[0], n * sizeof(double));
    }
    if (rank == size - 1) {
        // bottom boundary row (i==n-1) is cold
        memcpy(&buf0[(local_rows + 1) * n], &u[(n - 1) * n], n * sizeof(double));
    }

    double *curr = buf0;
    double *next = buf1;

    for (int it = 0; it < iters; ++it) {
        int up   = (rank == 0) ? MPI_PROC_NULL : rank - 1;
        int down = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

        // Exchange halo rows with neighbours.
        // Send first owned row to up, receive halo row from up.
        MPI_Sendrecv(&curr[1 * n], n, MPI_DOUBLE, up,   0,
                     &curr[0 * n], n, MPI_DOUBLE, up,   1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // Send last owned row to down, receive halo row from down.
        MPI_Sendrecv(&curr[local_rows * n], n, MPI_DOUBLE, down, 1,
                     &curr[(local_rows + 1) * n], n, MPI_DOUBLE, down, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Jacobi update for owned rows.
        for (long i = 1; i <= local_rows; ++i) {
            long global_i = lo + i - 1;
            if (global_i == 0 || global_i == n - 1) {
                // Fixed boundary rows: copy unchanged
                memcpy(&next[i * n], &curr[i * n], n * sizeof(double));
                continue;
            }
            // Fixed edge columns stay unchanged
            next[i * n + 0] = curr[i * n + 0];
            next[i * n + n - 1] = curr[i * n + n - 1];
            for (long j = 1; j < n - 1; ++j) {
                next[i * n + j] = 0.25 * (curr[(i - 1) * n + j] +
                                          curr[(i + 1) * n + j] +
                                          curr[i * n + (j - 1)] +
                                          curr[i * n + (j + 1)]);
            }
        }
        // Swap buffers for next iteration
        double *tmp = curr;
        curr = next;
        next = tmp;
    }

    // Gather the full grid on all ranks.
    if (n % size == 0) {
        int count = (int)(local_rows * n);
        MPI_Allgather(&curr[1 * n], count, MPI_DOUBLE,
                      u, count, MPI_DOUBLE, MPI_COMM_WORLD);
    } else {
        int *recvcounts = malloc(sizeof(int) * size);
        int *displs = malloc(sizeof(int) * size);
        for (int r = 0; r < size; ++r) {
            long rlo = n * r / size;
            long rhi = n * (r + 1) / size;
            recvcounts[r] = (int)((rhi - rlo) * n);
            displs[r] = (int)(rlo * n);
        }
        MPI_Allgatherv(&curr[1 * n], (int)(local_rows * n), MPI_DOUBLE,
                       u, recvcounts, displs, MPI_DOUBLE,
                       MPI_COMM_WORLD);
        free(recvcounts);
        free(displs);
    }

    free(buf0);
    free(buf1);
}

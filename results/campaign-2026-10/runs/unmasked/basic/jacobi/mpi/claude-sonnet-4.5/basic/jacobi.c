#include "jacobi.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

void jacobi(double *u, long n, int iters)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Decompose rows across ranks */
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    long nrows = hi - lo;

    /* Allocate local buffers: own rows + 2 halo rows (top and bottom) */
    long local_size = (nrows + 2) * n;
    double *local_u = (double *)malloc(local_size * sizeof(double));
    double *local_un = (double *)malloc(local_size * sizeof(double));

    /* Copy own rows from u to local_u (offset by 1 row for top halo) */
    for (long i = 0; i < nrows; i++) {
        memcpy(&local_u[(i + 1) * n], &u[(lo + i) * n], n * sizeof(double));
    }

    /* Determine neighbors */
    int top_neighbor = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int bottom_neighbor = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    /* Jacobi iterations */
    for (int iter = 0; iter < iters; iter++) {
        /* Exchange halo rows with neighbors */
        /* Send bottom row to bottom neighbor, receive top halo from top neighbor */
        MPI_Sendrecv(&local_u[nrows * n], n, MPI_DOUBLE, bottom_neighbor, 0,
                     &local_u[0], n, MPI_DOUBLE, top_neighbor, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Send top row to top neighbor, receive bottom halo from bottom neighbor */
        MPI_Sendrecv(&local_u[1 * n], n, MPI_DOUBLE, top_neighbor, 1,
                     &local_u[(nrows + 1) * n], n, MPI_DOUBLE, bottom_neighbor, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Copy to local_un to preserve boundaries */
        memcpy(local_un, local_u, local_size * sizeof(double));

        /* Update interior cells in own rows */
        for (long i = 1; i <= nrows; i++) {
            long global_i = lo + (i - 1);
            /* Skip if this is a boundary row globally */
            if (global_i == 0 || global_i == n - 1) continue;

            for (long j = 1; j < n - 1; j++) {
                long idx = i * n + j;
                local_un[idx] = 0.25 * (local_u[(i - 1) * n + j] +
                                        local_u[(i + 1) * n + j] +
                                        local_u[i * n + (j - 1)] +
                                        local_u[i * n + (j + 1)]);
            }
        }

        /* Swap buffers */
        double *tmp = local_u;
        local_u = local_un;
        local_un = tmp;
    }

    /* Gather all rows back to u on all ranks */
    int *recvcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));

    for (int r = 0; r < size; r++) {
        long r_lo = n * r / size;
        long r_hi = n * (r + 1) / size;
        recvcounts[r] = (int)((r_hi - r_lo) * n);
        displs[r] = (int)(r_lo * n);
    }

    MPI_Allgatherv(&local_u[1 * n], (int)(nrows * n), MPI_DOUBLE,
                   u, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

    free(recvcounts);
    free(displs);
    free(local_u);
    free(local_un);
}

#include "jacobi.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

void jacobi(double *u, long n, int iters)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Calculate row decomposition: each rank gets a contiguous block of rows */
    long total_rows = n - 2;  /* interior rows only (excluding top and bottom boundaries) */
    long base_rows = total_rows / size;
    long rem = total_rows % size;
    
    long local_rows = base_rows + (rank < rem ? 1 : 0);
    long row_start = 1 + base_rows * rank + (rank < rem ? rank : rem);

    /* Allocate local buffers with halo rows */
    /* We need space for: halo top (1 row) + local interior rows + halo bottom (1 row) */
    /* Plus the full n columns */
    long local_buf_rows = local_rows + 2;
    double *local_u = malloc(local_buf_rows * n * sizeof(double));
    double *local_un = malloc(local_buf_rows * n * sizeof(double));

    /* Initialize local buffers from the global grid */
    /* Copy the rows this rank is responsible for, plus boundaries if applicable */
    for (long i = 0; i < local_rows; i++) {
        long global_i = row_start + i;
        memcpy(&local_u[(i + 1) * n], &u[global_i * n], n * sizeof(double));
        memcpy(&local_un[(i + 1) * n], &u[global_i * n], n * sizeof(double));
    }

    /* Determine neighbors for halo exchange */
    int up_neighbor = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int down_neighbor = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    /* Temporary buffer for receiving halo data */
    double *halo_recv_top = malloc(n * sizeof(double));
    double *halo_recv_bottom = malloc(n * sizeof(double));

    for (int iter = 0; iter < iters; iter++) {
        /* Exchange halo rows with neighbors */
        /* Send top interior row up, receive into halo top from up neighbor */
        MPI_Sendrecv(&local_u[1 * n], n, MPI_DOUBLE, up_neighbor, 0,
                     halo_recv_top, n, MPI_DOUBLE, up_neighbor, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        /* Send bottom interior row down, receive into halo bottom from down neighbor */
        MPI_Sendrecv(&local_u[local_rows * n], n, MPI_DOUBLE, down_neighbor, 1,
                     halo_recv_bottom, n, MPI_DOUBLE, down_neighbor, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Copy received halo data into local buffers */
        if (rank > 0) {
            memcpy(&local_u[0 * n], halo_recv_top, n * sizeof(double));
        } else {
            /* Top boundary: copy from global u (row 0) */
            memcpy(&local_u[0 * n], &u[0 * n], n * sizeof(double));
        }
        
        if (rank < size - 1) {
            memcpy(&local_u[(local_rows + 1) * n], halo_recv_bottom, n * sizeof(double));
        } else {
            /* Bottom boundary: copy from global u (row n-1) */
            memcpy(&local_u[(local_rows + 1) * n], &u[(n - 1) * n], n * sizeof(double));
        }

        /* Perform Jacobi update for local interior rows */
        for (long li = 1; li <= local_rows; li++) {
            long global_i = row_start + li - 1;
            /* Skip left and right boundaries */
            for (long j = 1; j < n - 1; j++) {
                /* Only update if this is truly an interior cell (not on global boundaries) */
                if (global_i > 0 && global_i < n - 1) {
                    local_un[li * n + j] = 0.25 * (
                        local_u[(li - 1) * n + j] +
                        local_u[(li + 1) * n + j] +
                        local_u[li * n + (j - 1)] +
                        local_u[li * n + (j + 1)]
                    );
                }
            }
        }

        /* Swap buffers */
        double *tmp = local_u;
        local_u = local_un;
        local_un = tmp;
    }

    /* Gather results from all ranks to reconstruct the full grid on all ranks */
    /* Each rank contributes its local interior rows */
    
    /* First, prepare send buffer (just the interior rows, no halo) */
    double *sendbuf = malloc(local_rows * n * sizeof(double));
    for (long i = 0; i < local_rows; i++) {
        memcpy(&sendbuf[i * n], &local_u[(i + 1) * n], n * sizeof(double));
    }

    /* Prepare receive counts and displacements for Allgatherv */
    int *recvcounts = NULL;
    int *displs = NULL;
    
    if (rank == 0) {
        recvcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
    }
    
    int my_count = (int)(local_rows * n);
    MPI_Gather(&my_count, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; i++) {
            displs[i] = displs[i - 1] + recvcounts[i - 1];
        }
    }

    /* Gather all interior rows to rank 0 first */
    double *interior_grid = NULL;
    if (rank == 0) {
        interior_grid = malloc(total_rows * n * sizeof(double));
    }
    
    MPI_Gatherv(sendbuf, my_count, MPI_DOUBLE,
                interior_grid, recvcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    /* Broadcast the full grid to all ranks */
    if (rank == 0) {
        /* Copy interior rows back to u (preserving boundaries) */
        for (long i = 0; i < total_rows; i++) {
            memcpy(&u[(i + 1) * n], &interior_grid[i * n], n * sizeof(double));
        }
    }
    
    /* Broadcast the full u array to all ranks */
    MPI_Bcast(u, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Cleanup */
    free(local_u);
    free(local_un);
    free(halo_recv_top);
    free(halo_recv_bottom);
    free(sendbuf);
    free(recvcounts);
    free(displs);
    free(interior_grid);
}

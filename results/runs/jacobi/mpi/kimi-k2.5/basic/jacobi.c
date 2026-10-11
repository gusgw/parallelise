#include "jacobi.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

void jacobi(double *u, long n, int iters)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Calculate row decomposition: each rank gets a contiguous block of interior rows */
    long total_rows = n - 2;  /* Interior rows only (excluding top and bottom boundaries) */
    long rows_per_rank = total_rows / size;
    long remainder = total_rows % size;
    
    /* Each rank gets rows_per_rank or rows_per_rank+1 rows */
    long local_rows = rows_per_rank + (rank < remainder ? 1 : 0);
    
    /* Starting row in the global interior (0-based from row 1) */
    long row_start = rank * rows_per_rank + (rank < remainder ? rank : remainder);
    
    /* Actual global row indices (including boundary offset) */
    long global_start = 1 + row_start;  /* First interior row this rank owns */

    /* Allocate local buffers with halo rows */
    /* local_u[0] is halo from above, local_u[local_rows+1] is halo from below */
    long local_stride = n;
    double *local_u = malloc((local_rows + 2) * local_stride * sizeof(double));
    double *local_un = malloc((local_rows + 2) * local_stride * sizeof(double));

    /* Initialize local buffer from global u */
    /* Copy the rows this rank owns, plus halo boundaries */
    for (long i = 0; i < local_rows + 2; i++) {
        long global_i = global_start - 1 + i;  /* -1 for top halo row */
        if (global_i >= 0 && global_i < n) {
            memcpy(&local_u[i * local_stride], &u[global_i * n], n * sizeof(double));
            memcpy(&local_un[i * local_stride], &u[global_i * n], n * sizeof(double));
        }
    }

    /* Determine neighbors for halo exchange */
    int up_neighbor = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int down_neighbor = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    /* Jacobi iterations */
    for (int iter = 0; iter < iters; iter++) {
        /* Exchange halo rows with neighbors using MPI_Sendrecv_replace or separate calls */
        /* First: send to up, receive from down */
        if (up_neighbor != MPI_PROC_NULL || down_neighbor != MPI_PROC_NULL) {
            /* Send top real row up, receive bottom halo from down */
            MPI_Sendrecv(
                &local_u[1 * local_stride], (int)n, MPI_DOUBLE, up_neighbor, 0,
                &local_u[(local_rows + 1) * local_stride], (int)n, MPI_DOUBLE, down_neighbor, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );
            
            /* Send bottom real row down, receive top halo from up */
            MPI_Sendrecv(
                &local_u[local_rows * local_stride], (int)n, MPI_DOUBLE, down_neighbor, 1,
                &local_u[0 * local_stride], (int)n, MPI_DOUBLE, up_neighbor, 1,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );
        }

        /* Update interior cells */
        for (long li = 1; li <= local_rows; li++) {
            long global_i = global_start - 1 + li;
            /* Skip boundary rows - they should be fixed */
            if (global_i == 0 || global_i == n - 1) continue;
            
            for (long j = 1; j < n - 1; j++) {
                local_un[li * local_stride + j] = 0.25 * (
                    local_u[(li - 1) * local_stride + j] +
                    local_u[(li + 1) * local_stride + j] +
                    local_u[li * local_stride + (j - 1)] +
                    local_u[li * local_stride + (j + 1)]
                );
            }
        }

        /* Swap buffers */
        double *tmp = local_u;
        local_u = local_un;
        local_un = tmp;
    }

    /* Gather results back to all ranks */
    /* Each rank contributes its local rows (excluding halos) */
    double *sendbuf = malloc(local_rows * n * sizeof(double));
    for (long i = 0; i < local_rows; i++) {
        memcpy(&sendbuf[i * n], &local_u[(i + 1) * n], n * sizeof(double));
    }

    /* Gather counts and displacements */
    int *recvcounts = NULL;
    int *displs = NULL;
    if (rank == 0) {
        recvcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
    }
    
    int my_count = (int)(local_rows * n);
    MPI_Gather(&my_count, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        displs[0] = (int)n;  /* Start after top boundary row */
        for (int i = 1; i < size; i++) {
            displs[i] = displs[i - 1] + recvcounts[i - 1];
        }
    }

    /* Gather all interior rows to rank 0 */
    double *recvbuf = NULL;
    if (rank == 0) {
        recvbuf = malloc((n - 2) * n * sizeof(double));
    }
    
    MPI_Gatherv(sendbuf, my_count, MPI_DOUBLE,
                recvbuf, recvcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    /* Broadcast full grid to all ranks */
    if (rank == 0) {
        /* Copy interior rows to u */
        memcpy(&u[n], recvbuf, (n - 2) * n * sizeof(double));
        free(recvbuf);
        free(recvcounts);
        free(displs);
    }
    
    MPI_Bcast(u, (int)(n * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Cleanup */
    free(local_u);
    free(local_un);
    free(sendbuf);
}

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
    long total_rows = n - 2;  /* Interior rows only (excluding top and bottom boundaries) */
    long rows_per_rank = total_rows / size;
    long remainder = total_rows % size;
    
    long local_rows = rows_per_rank + (rank < remainder ? 1 : 0);
    long row_start = 1 + rank * rows_per_rank + (rank < remainder ? rank : remainder);
    long row_end = row_start + local_rows;  /* Exclusive */

    /* Allocate local buffers with halo rows */
    /* local_u has local_rows + 2 halo rows (top and bottom) */
    double *local_u = (double *)malloc((local_rows + 2) * n * sizeof(double));
    double *local_un = (double *)malloc((local_rows + 2) * n * sizeof(double));

    /* Initialize local buffers from global u */
    /* Copy the top halo row (row_start - 1) from global u */
    memcpy(&local_u[0], &u[(row_start - 1) * n], n * sizeof(double));
    /* Copy local interior rows */
    for (long i = 0; i < local_rows; i++) {
        memcpy(&local_u[(i + 1) * n], &u[(row_start + i) * n], n * sizeof(double));
    }
    /* Copy the bottom halo row (row_end) from global u */
    memcpy(&local_u[(local_rows + 1) * n], &u[row_end * n], n * sizeof(double));

    /* Copy to local_un for initial state */
    memcpy(local_un, local_u, (local_rows + 2) * n * sizeof(double));

    /* Determine neighbors */
    int up = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int down = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    /* Jacobi iterations */
    for (int iter = 0; iter < iters; iter++) {
        /* Exchange halo rows with neighbors */
        /* Send top interior row (local row 1) to up neighbor, receive into halo row 0 */
        MPI_Sendrecv(&local_u[1 * n], n, MPI_DOUBLE, up, 0,
                     &local_u[0 * n], n, MPI_DOUBLE, up, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        /* Send bottom interior row (local row local_rows) to down neighbor, receive into halo row local_rows+1 */
        MPI_Sendrecv(&local_u[local_rows * n], n, MPI_DOUBLE, down, 1,
                     &local_u[(local_rows + 1) * n], n, MPI_DOUBLE, down, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Update interior cells */
        for (long i = 1; i <= local_rows; i++) {
            for (long j = 1; j < n - 1; j++) {
                local_un[i * n + j] = 0.25 * (
                    local_u[(i - 1) * n + j] +
                    local_u[(i + 1) * n + j] +
                    local_u[i * n + (j - 1)] +
                    local_u[i * n + (j + 1)]
                );
            }
        }

        /* Swap buffers */
        double *temp = local_u;
        local_u = local_un;
        local_un = temp;
    }

    /* Gather results from all ranks to reconstruct full grid on all ranks */
    /* Each rank contributes its interior rows (excluding halos) */
    
    /* First, we need to gather the counts from each rank */
    int *recvcounts = NULL;
    int *displs = NULL;
    if (rank == 0) {
        recvcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
    }
    
    int local_count = (int)(local_rows * n);
    MPI_Gather(&local_count, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        displs[0] = n;  /* Start after top boundary row */
        for (int i = 1; i < size; i++) {
            displs[i] = displs[i - 1] + recvcounts[i - 1];
        }
    }
    
    /* Gather all interior rows to rank 0 */
    double *gather_buf = NULL;
    if (rank == 0) {
        gather_buf = (double *)malloc(n * n * sizeof(double));
        /* Copy boundary rows */
        memcpy(&gather_buf[0], &u[0], n * sizeof(double));  /* Top boundary */
        memcpy(&gather_buf[(n - 1) * n], &u[(n - 1) * n], n * sizeof(double));  /* Bottom boundary */
    }
    
    /* Gatherv to collect all interior rows */
    MPI_Gatherv(&local_u[1 * n], local_count, MPI_DOUBLE,
                gather_buf, recvcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    /* Broadcast the full grid to all ranks */
    if (rank == 0) {
        memcpy(u, gather_buf, n * n * sizeof(double));
    }
    MPI_Bcast(u, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Cleanup */
    free(local_u);
    free(local_un);
    if (rank == 0) {
        free(recvcounts);
        free(displs);
        free(gather_buf);
    }
}

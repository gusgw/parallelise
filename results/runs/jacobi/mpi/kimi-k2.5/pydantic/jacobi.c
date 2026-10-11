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
    long base_rows = total_rows / size;
    long rem = total_rows % size;
    
    /* This rank owns rows [row_start, row_end) of the interior (0-indexed within interior) */
    long local_rows = base_rows + (rank < rem ? 1 : 0);
    long row_start = rank * base_rows + (rank < rem ? rank : rem);
    long row_end = row_start + local_rows;
    
    /* Actual row indices in the full grid (interior rows are 1..n-2) */
    long global_row_start = 1 + row_start;
    long global_row_end = 1 + row_end;
    
    /* Allocate local buffers with halo rows */
    /* local_u contains: halo top (1 row) + local data + halo bottom (1 row) */
    long local_stride = n;
    double *local_u = malloc((local_rows + 2) * local_stride * sizeof(double));
    double *local_un = malloc((local_rows + 2) * local_stride * sizeof(double));
    
    /* Initialize local buffer from global u */
    /* Copy the top halo row (from rank-1 or boundary) and bottom halo row */
    for (long j = 0; j < n; j++) {
        /* Top halo: from global row global_row_start - 1 */
        local_u[j] = u[(global_row_start - 1) * n + j];
        /* Bottom halo: from global row global_row_end */
        local_u[(local_rows + 1) * n + j] = u[global_row_end * n + j];
    }
    
    /* Copy local rows */
    for (long i = 0; i < local_rows; i++) {
        memcpy(&local_u[(i + 1) * n], &u[(global_row_start + i) * n], n * sizeof(double));
    }
    
    memcpy(local_un, local_u, (local_rows + 2) * n * sizeof(double));
    
    /* Determine neighbors */
    int up = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int down = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;
    
    /* Jacobi iterations */
    for (int iter = 0; iter < iters; iter++) {
        /* Halo exchange: send my top data row up, receive into my top halo */
        /* Send row 1 (first local data row) up, receive into row 0 (top halo) */
        MPI_Sendrecv(&local_u[1 * n], n, MPI_DOUBLE, up, 0,
                     &local_u[0 * n], n, MPI_DOUBLE, up, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        /* Send my bottom data row down, receive into my bottom halo */
        /* Send row local_rows (last local data row) down, receive into row local_rows+1 */
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
        double *tmp = local_u;
        local_u = local_un;
        local_un = tmp;
    }
    
    /* Gather results: each rank has local_rows rows of interior data */
    /* Use MPI_Allgatherv to collect all interior rows to all ranks */
    
    /* Prepare send buffer (just the local data rows, no halos) */
    double *sendbuf = &local_u[1 * n];  /* Skip top halo */
    
    /* Prepare receive counts and displacements */
    int *recvcounts = NULL;
    int *displs = NULL;
    
    recvcounts = malloc(size * sizeof(int));
    displs = malloc(size * sizeof(int));
    
    for (int r = 0; r < size; r++) {
        long r_local_rows = base_rows + (r < rem ? 1 : 0);
        recvcounts[r] = (int)(r_local_rows * n);
    }
    
    displs[0] = n;  /* Start after the top boundary row */
    for (int r = 1; r < size; r++) {
        displs[r] = displs[r - 1] + recvcounts[r - 1];
    }
    
    /* Receive buffer is the interior of u (rows 1 to n-2) */
    double *recvbuf = &u[n];  /* Start at row 1 */
    
    MPI_Allgatherv(sendbuf, (int)(local_rows * n), MPI_DOUBLE,
                   recvbuf, recvcounts, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);
    
    free(recvcounts);
    free(displs);
    free(local_u);
    free(local_un);
}

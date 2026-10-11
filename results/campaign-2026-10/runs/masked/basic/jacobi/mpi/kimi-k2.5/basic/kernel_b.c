#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#include "kernel_b.h"
#include "problem.h"

void kernel_b(double *u, long n, int iters)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Calculate local row range: each rank gets a block of rows */
    long local_rows = n / size;
    long remainder = n % size;
    long row_start;
    
    if (rank < remainder) {
        local_rows++;
        row_start = rank * local_rows;
    } else {
        row_start = rank * local_rows + remainder;
    }

    /* Allocate local arrays with ghost rows for halo exchange */
    /* We need space for local_rows rows plus 2 ghost rows (top and bottom) */
    long local_size = (local_rows + 2) * n;
    double *local_u = malloc(sizeof(double) * local_size);
    double *local_un = malloc(sizeof(double) * local_size);
    
    if (!local_u || !local_un) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Initialize local arrays from global u */
    /* Copy row_start-1 to row_end (inclusive of ghost rows if they exist) */
    for (long i = 0; i < local_rows; i++) {
        for (long j = 0; j < n; j++) {
            local_u[(i + 1) * n + j] = u[(row_start + i) * n + j];
        }
    }

    /* Determine neighbors for halo exchange */
    int up_neighbor = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int down_neighbor = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    /* Jacobi iterations */
    for (int iter = 0; iter < iters; iter++) {
        /* Halo exchange: send top row up, receive from up; send bottom row down, receive from down */
        /* Send row 1 (first real row) to up neighbor, receive into row 0 (ghost row) from up */
        MPI_Sendrecv(&local_u[1 * n], n, MPI_DOUBLE, up_neighbor, 0,
                     &local_u[0 * n], n, MPI_DOUBLE, up_neighbor, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        /* Send row local_rows (last real row) to down neighbor, receive into row local_rows+1 (ghost row) from down */
        MPI_Sendrecv(&local_u[local_rows * n], n, MPI_DOUBLE, down_neighbor, 1,
                     &local_u[(local_rows + 1) * n], n, MPI_DOUBLE, down_neighbor, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Perform Jacobi update for local interior points */
        for (long i = 1; i <= local_rows; i++) {
            long global_i = row_start + (i - 1);
            /* Skip boundary rows in global grid */
            if (global_i == 0 || global_i == n - 1) {
                /* Copy boundary values unchanged */
                for (long j = 0; j < n; j++) {
                    local_un[i * n + j] = local_u[i * n + j];
                }
                continue;
            }
            /* Update interior columns */
            for (long j = 1; j < n - 1; j++) {
                local_un[i * n + j] = 0.25 * (
                    local_u[(i - 1) * n + j] +
                    local_u[(i + 1) * n + j] +
                    local_u[i * n + (j - 1)] +
                    local_u[i * n + (j + 1)]
                );
            }
            /* Copy boundary columns unchanged */
            local_un[i * n + 0] = local_u[i * n + 0];
            local_un[i * n + (n - 1)] = local_u[i * n + (n - 1)];
        }

        /* Swap pointers */
        double *temp = local_u;
        local_u = local_un;
        local_un = temp;
    }

    /* Gather results back to rank 0 */
    /* First, gather the counts from each rank */
    int *recvcounts = NULL;
    int *displs = NULL;
    
    if (rank == 0) {
        recvcounts = malloc(sizeof(int) * size);
        displs = malloc(sizeof(int) * size);
    }
    
    int local_count = (int)(local_rows * n);
    MPI_Gather(&local_count, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; i++) {
            displs[i] = displs[i - 1] + recvcounts[i - 1];
        }
    }
    
    /* Gather the actual data (skip ghost rows) */
    MPI_Gatherv(&local_u[1 * n], local_count, MPI_DOUBLE,
                u, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    /* Broadcast the full array to all ranks */
    MPI_Bcast(u, (int)(n * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Clean up */
    free(local_u);
    free(local_un);
    if (rank == 0) {
        free(recvcounts);
        free(displs);
    }
}

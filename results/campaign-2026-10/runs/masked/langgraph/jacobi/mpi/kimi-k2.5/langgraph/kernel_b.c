#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kernel_b.h"

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

    /* Allocate local grid with halo rows (2 extra rows for ghost cells) */
    double *local_u = malloc(sizeof(double) * (local_rows + 2) * n);
    double *local_un = malloc(sizeof(double) * (local_rows + 2) * n);
    
    if (!local_u || !local_un) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Copy initial data from global u to local_u (excluding halo rows) */
    /* local_u[1..local_rows] corresponds to u[row_start..row_end-1] */
    for (long i = 0; i < local_rows; i++) {
        memcpy(&local_u[(i + 1) * n], &u[(row_start + i) * n], sizeof(double) * n);
    }

    /* Determine neighbors */
    int up = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int down = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    /* Jacobi iterations */
    for (int iter = 0; iter < iters; iter++) {
        /* Halo exchange: send boundary rows to neighbors */
        /* 
         * Exchange with up neighbor:
         * - Send my first real row (local_u[1]) to up neighbor
         * - Receive into my halo row 0 from up neighbor
         */
        MPI_Sendrecv(&local_u[1 * n], n, MPI_DOUBLE, up, 0,
                     &local_u[0 * n], n, MPI_DOUBLE, up, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        /* 
         * Exchange with down neighbor:
         * - Send my last real row (local_u[local_rows]) to down neighbor
         * - Receive into my last halo row from down neighbor
         */
        MPI_Sendrecv(&local_u[local_rows * n], n, MPI_DOUBLE, down, 0,
                     &local_u[(local_rows + 1) * n], n, MPI_DOUBLE, down, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Compute new values for interior points */
        /* Iterate over local rows */
        for (long i = 0; i < local_rows; i++) {
            long global_i = row_start + i;
            
            /* Skip global boundary rows (i=0 and i=n-1) - they stay fixed */
            if (global_i == 0 || global_i == n - 1) {
                /* Copy boundary row unchanged */
                for (long j = 0; j < n; j++) {
                    local_un[(i + 1) * n + j] = local_u[(i + 1) * n + j];
                }
                continue;
            }
            
            /* For interior rows, update interior columns */
            for (long j = 0; j < n; j++) {
                if (j == 0 || j == n - 1) {
                    /* Boundary columns remain unchanged */
                    local_un[(i + 1) * n + j] = local_u[(i + 1) * n + j];
                } else {
                    /* Interior point: apply Jacobi stencil */
                    /* local_u[i+1] is current row, local_u[i] is up, local_u[i+2] is down */
                    local_un[(i + 1) * n + j] = 0.25 * (
                        local_u[i * n + j] +             /* up */
                        local_u[(i + 2) * n + j] +       /* down */
                        local_u[(i + 1) * n + (j - 1)] + /* left */
                        local_u[(i + 1) * n + (j + 1)]   /* right */
                    );
                }
            }
        }

        /* Swap pointers */
        double *temp = local_u;
        local_u = local_un;
        local_un = temp;
    }

    /* Copy final data back to global u */
    for (long i = 0; i < local_rows; i++) {
        memcpy(&u[(row_start + i) * n], &local_u[(i + 1) * n], sizeof(double) * n);
    }

    free(local_u);
    free(local_un);
}

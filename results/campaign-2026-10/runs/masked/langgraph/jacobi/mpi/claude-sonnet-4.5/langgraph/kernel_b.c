#include "kernel_b.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

void kernel_b(double *u, long n, int iters)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Decompose rows across ranks
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    long local_rows = hi - lo;
    
    // Allocate local buffer with halo rows
    double *local_u = malloc(sizeof(double) * (local_rows + 2) * n);
    double *local_un = malloc(sizeof(double) * (local_rows + 2) * n);
    
    // Copy local rows from global array (offset by 1 for halo)
    for (long i = 0; i < local_rows; i++) {
        for (long j = 0; j < n; j++) {
            local_u[(i + 1) * n + j] = u[(lo + i) * n + j];
        }
    }
    
    // Determine neighbors
    int prev_rank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int next_rank = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;
    
    // Jacobi iterations
    for (int iter = 0; iter < iters; iter++) {
        // Exchange halo rows
        // Send top row to prev, receive from next into bottom halo
        MPI_Sendrecv(&local_u[1 * n], n, MPI_DOUBLE, prev_rank, 0,
                     &local_u[(local_rows + 1) * n], n, MPI_DOUBLE, next_rank, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Send bottom row to next, receive from prev into top halo
        MPI_Sendrecv(&local_u[local_rows * n], n, MPI_DOUBLE, next_rank, 1,
                     &local_u[0], n, MPI_DOUBLE, prev_rank, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Copy current state
        memcpy(local_un, local_u, sizeof(double) * (local_rows + 2) * n);
        
        // Update interior points
        for (long i = 1; i <= local_rows; i++) {
            long global_i = lo + (i - 1);
            // Skip boundary rows and columns
            if (global_i == 0 || global_i == n - 1) continue;
            
            for (long j = 1; j < n - 1; j++) {
                local_un[i * n + j] = 0.25 * (
                    local_u[(i - 1) * n + j] +
                    local_u[(i + 1) * n + j] +
                    local_u[i * n + (j - 1)] +
                    local_u[i * n + (j + 1)]
                );
            }
        }
        
        // Swap buffers
        double *temp = local_u;
        local_u = local_un;
        local_un = temp;
    }
    
    // Gather results back to all ranks
    // First, extract local data (without halos)
    double *send_buf = malloc(sizeof(double) * local_rows * n);
    for (long i = 0; i < local_rows; i++) {
        for (long j = 0; j < n; j++) {
            send_buf[i * n + j] = local_u[(i + 1) * n + j];
        }
    }
    
    // Compute send counts and displacements
    int *recvcounts = malloc(sizeof(int) * size);
    int *displs = malloc(sizeof(int) * size);
    for (int r = 0; r < size; r++) {
        long r_lo = n * r / size;
        long r_hi = n * (r + 1) / size;
        recvcounts[r] = (r_hi - r_lo) * n;
        displs[r] = r_lo * n;
    }
    
    // Gather to all ranks
    MPI_Allgatherv(send_buf, local_rows * n, MPI_DOUBLE,
                   u, recvcounts, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);
    
    free(send_buf);
    free(recvcounts);
    free(displs);
    free(local_u);
    free(local_un);
}

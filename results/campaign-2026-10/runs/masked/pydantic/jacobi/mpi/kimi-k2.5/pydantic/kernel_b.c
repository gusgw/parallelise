#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kernel_b.h"
#include "problem.h"

void kernel_b(double *u, long n, int iters)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Calculate local row range for this rank (row-wise decomposition) */
    long local_rows = n / size;
    long remainder = n % size;
    long row_start;
    
    if (rank < remainder) {
        local_rows++;
        row_start = rank * local_rows;
    } else {
        row_start = rank * local_rows + remainder;
    }

    /* Allocate working arrays */
    double *local_u = malloc(sizeof(double) * local_rows * n);
    double *full_u = malloc(sizeof(double) * n * n);
    double *full_un = malloc(sizeof(double) * n * n);
    
    if (!local_u || !full_u || !full_un) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Copy initial data from global u to local_u */
    memcpy(local_u, &u[row_start * n], sizeof(double) * local_rows * n);

    /* Gather counts and displacements for Allgatherv */
    int *recvcounts = malloc(sizeof(int) * size);
    int *displs = malloc(sizeof(int) * size);
    
    if (!recvcounts || !displs) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    int mycount = (int)(local_rows * n);
    MPI_Allgather(&mycount, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);
    
    displs[0] = 0;
    for (int i = 1; i < size; i++) {
        displs[i] = displs[i-1] + recvcounts[i-1];
    }

    /* Jacobi iterations */
    for (int iter = 0; iter < iters; iter++) {
        /* Gather full grid to all ranks */
        MPI_Allgatherv(local_u, (int)(local_rows * n), MPI_DOUBLE,
                       full_u, recvcounts, displs, MPI_DOUBLE,
                       MPI_COMM_WORLD);
        
        /* Compute new values for interior points */
        memcpy(full_un, full_u, sizeof(double) * n * n);
        
        for (long i = 1; i < n - 1; i++) {
            for (long j = 1; j < n - 1; j++) {
                full_un[i * n + j] = 0.25 * (full_u[(i - 1) * n + j] +
                                              full_u[(i + 1) * n + j] +
                                              full_u[i * n + j - 1] +
                                              full_u[i * n + j + 1]);
            }
        }
        
        /* Extract local portion for next iteration */
        memcpy(local_u, &full_un[row_start * n], sizeof(double) * local_rows * n);
    }

    /* Copy final data back to global u */
    memcpy(&u[row_start * n], local_u, sizeof(double) * local_rows * n);

    /* Clean up */
    free(local_u);
    free(full_u);
    free(full_un);
    free(recvcounts);
    free(displs);
}

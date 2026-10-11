#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include "jacobi.h"

void jacobi(double *u, long n, int iters)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* The grid has fixed boundaries at row 0 and row n-1.
       Interior rows that need updating are 1 to n-2. */
    long total_interior = n - 2;
    
    /* Determine which interior rows this rank owns.
       We decompose the interior rows [1, n-1) across ranks. */
    long lo = 1 + (total_interior * rank) / size;
    long hi = 1 + (total_interior * (rank + 1)) / size;
    long local_rows = hi - lo;

    /* Allocate local buffers with halo rows.
       We need space for local_rows plus up to 2 halo rows (top and bottom). */
    long local_buf_rows = local_rows + 2;  /* +2 for halo rows */
    double *local_u = malloc(sizeof(double) * local_buf_rows * n);
    double *local_unew = malloc(sizeof(double) * local_buf_rows * n);

    /* Copy this rank's rows from the global grid to local buffer.
       local_u[1..local_rows] corresponds to global rows [lo..hi). */
    for (long i = 0; i < local_rows; i++) {
        memcpy(&local_u[(i + 1) * n], &u[(lo + i) * n], sizeof(double) * n);
    }

    /* Initialize halo rows for boundary ranks.
       If rank == 0, copy the top boundary (row 0) into local_u[0].
       If rank == size-1, copy the bottom boundary (row n-1) into local_u[local_rows+1]. */
    if (rank == 0) {
        memcpy(&local_u[0], &u[0], sizeof(double) * n);
    }
    if (rank == size - 1) {
        memcpy(&local_u[(local_rows + 1) * n], &u[(n - 1) * n], sizeof(double) * n);
    }

    /* Determine neighbors */
    int up_neighbor = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int down_neighbor = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    /* Jacobi iterations */
    for (int iter = 0; iter < iters; iter++) {
        /* Exchange halo rows with neighbors.
           Send my top row (local_u[1]) to up neighbor, receive into local_u[0].
           Send my bottom row (local_u[local_rows]) to down neighbor, receive into local_u[local_rows+1]. */
        MPI_Sendrecv(&local_u[1 * n], n, MPI_DOUBLE, up_neighbor, 0,
                     &local_u[0], n, MPI_DOUBLE, up_neighbor, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        MPI_Sendrecv(&local_u[local_rows * n], n, MPI_DOUBLE, down_neighbor, 1,
                     &local_u[(local_rows + 1) * n], n, MPI_DOUBLE, down_neighbor, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Update interior cells.
           For each local row i (1 to local_rows), update columns 1 to n-2. */
        for (long i = 1; i <= local_rows; i++) {
            for (long j = 1; j < n - 1; j++) {
                long idx = i * n + j;
                local_unew[idx] = 0.25 * (local_u[(i - 1) * n + j] +
                                          local_u[(i + 1) * n + j] +
                                          local_u[i * n + (j - 1)] +
                                          local_u[i * n + (j + 1)]);
            }
        }

        /* Copy boundary columns (j=0 and j=n-1) from old to new buffer
           to ensure they remain unchanged. */
        for (long i = 1; i <= local_rows; i++) {
            local_unew[i * n] = local_u[i * n];
            local_unew[i * n + (n - 1)] = local_u[i * n + (n - 1)];
        }

        /* Copy halo rows to new buffer so they're preserved after swap.
           The halo rows are local_unew[0] and local_unew[local_rows+1]. */
        memcpy(&local_unew[0], &local_u[0], sizeof(double) * n);
        memcpy(&local_unew[(local_rows + 1) * n], &local_u[(local_rows + 1) * n], sizeof(double) * n);

        /* Swap buffers */
        double *tmp = local_u;
        local_u = local_unew;
        local_unew = tmp;
    }

    /* Copy local results back to global grid (only interior columns).
       The boundary columns (0 and n-1) should remain unchanged. */
    for (long i = 0; i < local_rows; i++) {
        memcpy(&u[(lo + i) * n + 1], &local_u[(i + 1) * n + 1], sizeof(double) * (n - 2));
    }

    /* Gather the full grid on all ranks using Allgatherv.
       Each rank contributes its interior rows. */
    int *recvcounts = malloc(sizeof(int) * size);
    int *displs = malloc(sizeof(int) * size);

    /* Each rank computes recvcounts and displs for all ranks */
    for (int r = 0; r < size; r++) {
        long r_lo = 1 + (total_interior * r) / size;
        long r_hi = 1 + (total_interior * (r + 1)) / size;
        long r_rows = r_hi - r_lo;
        recvcounts[r] = (int)(r_rows * n);
        displs[r] = (int)(r_lo * n);
    }

    int my_count = (int)(local_rows * n);

    /* Allgatherv to assemble the full grid on all ranks */
    MPI_Allgatherv(&u[lo * n], my_count, MPI_DOUBLE,
                   u, recvcounts, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    free(local_u);
    free(local_unew);
    free(recvcounts);
    free(displs);
}

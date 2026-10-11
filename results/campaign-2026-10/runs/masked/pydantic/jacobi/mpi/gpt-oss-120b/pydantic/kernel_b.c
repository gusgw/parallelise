#include "kernel_b.h"
#include "problem.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

void kernel_b(double *u, long n, int iters) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Determine the range of rows this rank owns: [lo, hi)
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    long local_n = hi - lo; // number of owned rows

    // Allocate current and next grids with two halo rows (top and bottom)
    double *curr = (double *)malloc((local_n + 2) * n * sizeof(double));
    double *next = (double *)malloc((local_n + 2) * n * sizeof(double));
    if (!curr || !next) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Initialise interior rows (including possible global boundary rows)
    for (long i_local = 1; i_local <= local_n; ++i_local) {
        long i_global = lo + i_local - 1;
        for (long j = 0; j < n; ++j) {
            curr[i_local * n + j] = u_init(i_global, j, n);
        }
    }
    // Initialise halo rows to zero (they will be overwritten by communication)
    memset(curr, 0, n * sizeof(double));                                 // top halo (row 0)
    memset(curr + (local_n + 1) * n, 0, n * sizeof(double));               // bottom halo

    // Main Jacobi iterations
    for (int it = 0; it < iters; ++it) {
        // Exchange halo rows with neighbours using MPI_PROC_NULL for missing neighbours
        int up   = (rank == 0)     ? MPI_PROC_NULL : rank - 1;
        int down = (rank == size-1)? MPI_PROC_NULL : rank + 1;

        // Send my first interior row upward, receive into top halo
        MPI_Sendrecv(curr + 1 * n, n, MPI_DOUBLE, up,   0,
                     curr + 0 * n, n, MPI_DOUBLE, up,   0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // Send my last interior row downward, receive into bottom halo
        MPI_Sendrecv(curr + local_n * n, n, MPI_DOUBLE, down, 0,
                     curr + (local_n + 1) * n, n, MPI_DOUBLE, down, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Compute new values for interior rows (skip global boundaries)
        for (long i_local = 1; i_local <= local_n; ++i_local) {
            long i_global = lo + i_local - 1;
            if (i_global == 0 || i_global == n - 1) {
                // Global boundary rows stay unchanged (Dirichlet)
                memcpy(next + i_local * n, curr + i_local * n, n * sizeof(double));
                continue;
            }
            // First and last column are also Dirichlet (remain unchanged)
            next[i_local * n + 0] = curr[i_local * n + 0];
            next[i_local * n + (n - 1)] = curr[i_local * n + (n - 1)];
            for (long j = 1; j < n - 1; ++j) {
                next[i_local * n + j] = 0.25 * (curr[(i_local - 1) * n + j] +
                                             curr[(i_local + 1) * n + j] +
                                             curr[i_local * n + (j - 1)] +
                                             curr[i_local * n + (j + 1)]);
            }
        }
        // Swap pointers for next iteration
        double *tmp = curr;
        curr = next;
        next = tmp;
    }

    // Gather the final grid onto all ranks using Allgatherv (interior rows only)
    int *recvcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    int local_count = (int)(local_n * n);
    MPI_Allgather(&local_count, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);
    displs[0] = 0;
    for (int i = 1; i < size; ++i) {
        displs[i] = displs[i - 1] + recvcounts[i - 1];
    }
    MPI_Allgatherv(curr + 1 * n, local_count, MPI_DOUBLE,
                   u, recvcounts, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    free(curr);
    free(next);
    free(recvcounts);
    free(displs);
}

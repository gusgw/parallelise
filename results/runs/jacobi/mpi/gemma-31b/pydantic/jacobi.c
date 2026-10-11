#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include "jacobi.h"

void jacobi(double *u, long n, int iters) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    long local_n = hi - lo;

    double *u_local = (double *)malloc((local_n + 2) * n * sizeof(double));
    double *u_next = (double *)malloc((local_n + 2) * n * sizeof(double));

    // Initial fill of local buffers from the global u
    // We need to include halo rows (lo-1 and hi)
    // Handle boundaries carefully
    for (long i = 0; i < local_n; i++) {
        memcpy(&u_local[(i + 1) * n], &u[(lo + i) * n], n * sizeof(double));
        memcpy(&u_next[(i + 1) * n], &u[(lo + i) * n], n * sizeof(double));
    }

    int up = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int down = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    for (int iter = 0; iter < iters; iter++) {
        // Halo exchange
        MPI_Sendrecv(&u_local[1 * n], n, MPI_DOUBLE, up, 0,
                     &u_local[(local_n + 1) * n], n, MPI_DOUBLE, down, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&u_local[local_n * n], n, MPI_DOUBLE, down, 1,
                     &u_local[0 * n], n, MPI_DOUBLE, up, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Compute interior
        for (long i = 0; i < local_n; i++) {
            long global_i = lo + i;
            // Skip if it's a global boundary row
            if (global_i == 0 || global_i == n - 1) continue;

            for (long j = 1; j < n - 1; j++) {
                u_next[(i + 1) * n + j] = 0.25 * (
                    u_local[i * n + j] + 
                    u_local[(i + 2) * n + j] + 
                    u_local[(i + 1) * n + (j - 1)] + 
                    u_local[(i + 1) * n + (j + 1)]
                );
            }
        }
        // Swap buffers
        double *tmp = u_local;
        u_local = u_next;
        u_next = tmp;
    }

    // Gather results back to u on all ranks
    // Each rank has its slice in u_local[1...local_n]
    // We use MPI_Allgather to collect all slices. 
    // Since slices can be different sizes, we use MPI_Allgatherv.
    long *recv_counts = (long *)malloc(size * sizeof(long));
    long *displs = (long *)malloc(size * sizeof(long));
    for (int i = 0; i < size; i++) {
        long l = n * i / size;
        long h = n * (i + 1) / size;
        recv_counts[i] = (h - l) * n;
        displs[i] = l * n;
    }

    MPI_Allgatherv(&u_local[1 * n], (local_n) * n, MPI_DOUBLE,
                   u, recv_counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

    free(u_local);
    free(u_next);
    free(recv_counts);
    free(displs);
}

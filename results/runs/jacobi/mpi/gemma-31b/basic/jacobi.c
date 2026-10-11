#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include "jacobi.h"

void jacobi(double *u, long n, int iters) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long lo = (n * rank) / size;
    long hi = (n * (rank + 1)) / size;
    long local_rows = hi - lo;

    double *u_local = (double *)malloc((local_rows + 2) * n * sizeof(double));
    double *u_next = (double *)malloc((local_rows + 2) * n * sizeof(double));

    // Initialize local buffers
    // u_local[0...n-1] is top halo, [n...n + local_rows*n - 1] is data, [n + local_rows*n... (local_rows+2)*n-1] is bottom halo
    memcpy(u_local + n, u + lo * n, local_rows * n * sizeof(double));
    memcpy(u_next + n, u + lo * n, local_rows * n * sizeof(double));

    int up = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int down = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    for (int iter = 0; iter < iters; iter++) {
        // Halo exchange
        // Send first data row to up, receive into top halo
        MPI_Sendrecv(u_local + n, n, MPI_DOUBLE, up, 0,
                     u_local, n, MPI_DOUBLE, up, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // Send last data row to down, receive into bottom halo
        MPI_Sendrecv(u_local + (local_rows) * n, n, MPI_DOUBLE, down, 1,
                     u_local + (local_rows + 1) * n, n, MPI_DOUBLE, down, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Compute interior
        for (long i = 1; i <= local_rows; i++) {
            long global_i = lo + i - 1;
            if (global_i > 0 && global_i < n - 1) {
                for (long j = 1; j < n - 1; j++) {
                    u_next[i * n + j] = 0.25 * (u_local[(i - 1) * n + j] + 
                                               u_local[(i + 1) * n + j] + 
                                               u_local[i * n + (j - 1)] + 
                                               u_local[i * n + (j + 1)]);
                }
            }
        }
        
        // Swap buffers
        double *tmp = u_local;
        u_local = u_next;
        u_next = tmp;
    }

    // Final result is in u_local (after the last swap)
    int *recv_counts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        long r_lo = (n * i) / size;
        long r_hi = (n * (i + 1)) / size;
        recv_counts[i] = (int)((r_hi - r_lo) * n);
        displs[i] = (int)(r_lo * n);
    }

    MPI_Allgatherv(u_local + n, local_rows * n, MPI_DOUBLE,
                   u, recv_counts, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    free(u_local);
    free(u_next);
    free(recv_counts);
    free(displs);
}

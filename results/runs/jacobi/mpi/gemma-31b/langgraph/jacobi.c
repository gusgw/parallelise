#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "jacobi.h"

void jacobi(double *u, long n, int iters) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long lo = (n * rank) / size;
    long hi = (n * (rank + 1)) / size;
    long local_n = hi - lo;

    double *u_local = (double *)malloc((local_n + 2) * n * sizeof(double));
    double *u_next = (double *)malloc((local_n + 2) * n * sizeof(double));

    // Initial fill: copy the slice from u into u_local (with halo space)
    // u_local[0] is top halo, u_local[1..local_n] are actual rows, u_local[local_n+1] is bottom halo
    memcpy(&u_local[n], &u[lo * n], local_n * n * sizeof(double));

    int up = rank - 1;
    int down = rank + 1;
    if (rank == 0) up = MPI_PROC_NULL;
    if (rank == size - 1) down = MPI_PROC_NULL;

    for (int it = 0; it < iters; it++) {
        // Halo exchange
        // Send my first actual row (u_local[n]) to up, receive into bottom halo (u_local[(local_n+1)*n])
        MPI_Sendrecv(&u_local[n], n, MPI_DOUBLE, up, 0,
                     &u_local[(local_n + 1) * n], n, MPI_DOUBLE, down, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // Send my last actual row (u_local[local_n*n]) to down, receive into top halo (u_local[0])
        MPI_Sendrecv(&u_local[local_n * n], n, MPI_DOUBLE, down, 1,
                     &u_local[0], n, MPI_DOUBLE, up, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Compute interior cells
        for (long i = 1; i <= local_n; i++) {
            long global_i = lo + i - 1;
            if (global_i == 0 || global_i == n - 1) {
                // Boundary rows: copy as is
                memcpy(&u_next[i * n], &u_local[i * n], n * sizeof(double));
                continue;
            }
            for (long j = 0; j < n; j++) {
                if (j == 0 || j == n - 1) {
                    // Boundary columns: copy as is
                    u_next[i * n + j] = u_local[i * n + j];
                } else {
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

    // Gather the results back to u on all ranks
    double *send_buf = (double *)malloc(local_n * n * sizeof(double));
    memcpy(send_buf, &u_local[n], local_n * n * sizeof(double));

    int *recv_counts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    
    for (int r = 0; r < size; r++) {
        long r_lo = (n * r) / size;
        long r_hi = (n * (r + 1)) / size;
        recv_counts[r] = (int)((r_hi - r_lo) * n);
        displs[r] = (int)(r_lo * n);
    }

    MPI_Allgatherv(send_buf, local_n * n, MPI_DOUBLE,
                   u, recv_counts, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    free(u_local);
    free(u_next);
    free(send_buf);
    free(recv_counts);
    free(displs);
}

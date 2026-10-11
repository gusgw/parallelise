#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "reference.h"
#include "problem.h"

void jacobi_debug(double *u, long n, int iters)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long total_interior = n - 2;
    long lo = 1 + (total_interior * rank) / size;
    long hi = 1 + (total_interior * (rank + 1)) / size;
    long local_rows = hi - lo;

    long local_buf_rows = local_rows + 2;
    double *local_u = malloc(sizeof(double) * local_buf_rows * n);
    double *local_unew = malloc(sizeof(double) * local_buf_rows * n);

    for (long i = 0; i < local_rows; i++) {
        memcpy(&local_u[(i + 1) * n], &u[(lo + i) * n], sizeof(double) * n);
    }

    if (rank == 0) {
        memcpy(&local_u[0], &u[0], sizeof(double) * n);
    }
    if (rank == size - 1) {
        memcpy(&local_u[(local_rows + 1) * n], &u[(n - 1) * n], sizeof(double) * n);
    }

    int up_neighbor = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int down_neighbor = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    if (rank == 0) {
        printf("Before iterations:\n");
        printf("  local_u[0][0] = %f (should be 1.0, top boundary)\n", local_u[0]);
        printf("  local_u[1][0] = %f (should be 1.0, row 1 col 0)\n", local_u[n]);
        printf("  local_u[1][1] = %f (should be 0.0, row 1 col 1)\n", local_u[n + 1]);
    }

    for (int iter = 0; iter < iters; iter++) {
        MPI_Sendrecv(&local_u[1 * n], n, MPI_DOUBLE, up_neighbor, 0,
                     &local_u[0], n, MPI_DOUBLE, up_neighbor, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        MPI_Sendrecv(&local_u[local_rows * n], n, MPI_DOUBLE, down_neighbor, 1,
                     &local_u[(local_rows + 1) * n], n, MPI_DOUBLE, down_neighbor, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (long i = 1; i <= local_rows; i++) {
            for (long j = 1; j < n - 1; j++) {
                long idx = i * n + j;
                local_unew[idx] = 0.25 * (local_u[(i - 1) * n + j] +
                                          local_u[(i + 1) * n + j] +
                                          local_u[i * n + (j - 1)] +
                                          local_u[i * n + (j + 1)]);
            }
        }

        for (long i = 1; i <= local_rows; i++) {
            local_unew[i * n] = local_u[i * n];
            local_unew[i * n + (n - 1)] = local_u[i * n + (n - 1)];
        }

        double *tmp = local_u;
        local_u = local_unew;
        local_unew = tmp;
        
        if (rank == 0 && iter == 0) {
            printf("After iter 0:\n");
            printf("  local_u[0][0] = %f (should be 1.0, top boundary)\n", local_u[0]);
            printf("  local_u[1][0] = %f (should be 1.0, row 1 col 0)\n", local_u[n]);
            printf("  local_u[1][1] = %f (should be 0.25, row 1 col 1 after update)\n", local_u[n + 1]);
        }
    }

    for (long i = 0; i < local_rows; i++) {
        memcpy(&u[(lo + i) * n + 1], &local_u[(i + 1) * n + 1], sizeof(double) * (n - 2));
    }

    int *recvcounts = malloc(sizeof(int) * size);
    int *displs = malloc(sizeof(int) * size);

    for (int r = 0; r < size; r++) {
        long r_lo = 1 + (total_interior * r) / size;
        long r_hi = 1 + (total_interior * (r + 1)) / size;
        long r_rows = r_hi - r_lo;
        recvcounts[r] = (int)(r_rows * n);
        displs[r] = (int)(r_lo * n);
    }

    int my_count = (int)(local_rows * n);

    MPI_Allgatherv(&u[lo * n], my_count, MPI_DOUBLE,
                   u, recvcounts, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    free(local_u);
    free(local_unew);
    free(recvcounts);
    free(displs);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long n = REF_N;
    int iters = (int)REF_ITERS;

    double *u = malloc(sizeof(double) * n * n);
    if (!u) {
        fprintf(stderr, "alloc failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            u[i * n + j] = u_init(i, j, n);
        }
    }

    jacobi_debug(u, n, iters);

    if (rank == 0) {
        double cs = checksum(u, n);
        printf("Final checksum: %.12f\n", cs);
        printf("Expected: %.12f\n", REF_CHECKSUM);
    }

    free(u);
    MPI_Finalize();
    return 0;
}

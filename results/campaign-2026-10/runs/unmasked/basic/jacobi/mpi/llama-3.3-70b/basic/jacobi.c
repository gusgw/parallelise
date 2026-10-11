#include "jacobi.h"
#include "reference.h"
#include <mpi.h>
#include <stdlib.h>
#include <math.h>

void jacobi(double *u, long n, int iters) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;

    double *u_new = malloc(n * n * sizeof(double));

    for (int iter = 0; iter < iters; iter++) {
        // Halo exchange
        MPI_Request req[4];
        MPI_Status stat[4];
        if (rank > 0) {
            MPI_Irecv(&u_new[lo * n], n, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &req[0]);
        }
        if (rank < size - 1) {
            MPI_Isend(&u[hi * n - n], n, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &req[1]);
        }
        if (rank > 0) {
            MPI_Irecv(&u_new[(lo - 1) * n], n, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &req[2]);
        }
        if (rank < size - 1) {
            MPI_Isend(&u[hi * n], n, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &req[3]);
        }

        // Update interior cells
        for (long i = lo; i < hi; i++) {
            for (long j = 1; j < n - 1; j++) {
                u_new[i * n + j] = 0.25 * (u[(i - 1) * n + j] + u[(i + 1) * n + j] + u[i * n + j - 1] + u[i * n + j + 1]);
            }
        }

        // Wait for halo exchange to complete
        if (rank > 0) {
            MPI_Wait(&req[0], &stat[0]);
        }
        if (rank < size - 1) {
            MPI_Wait(&req[1], &stat[1]);
        }
        if (rank > 0) {
            MPI_Wait(&req[2], &stat[2]);
        }
        if (rank < size - 1) {
            MPI_Wait(&req[3], &stat[3]);
        }

        // Swap u and u_new
        double *tmp = u;
        u = u_new;
        u_new = tmp;
    }

    free(u_new);
}

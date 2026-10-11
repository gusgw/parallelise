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

    double *u_new = (double *) malloc(n * n * sizeof(double));

    for (int iter = 0; iter < iters; iter++) {
        // halo exchange
        double *u_halo = (double *) malloc(2 * n * sizeof(double));
        if (rank > 0) {
            MPI_Sendrecv(u + (lo - 1) * n, n, MPI_DOUBLE, rank - 1, 0,
                          u_halo, n, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < size - 1) {
            MPI_Sendrecv(u + hi * n, n, MPI_DOUBLE, rank + 1, 0,
                          u_halo + n, n, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // update interior cells
        for (long i = lo; i < hi; i++) {
            for (long j = 1; j < n - 1; j++) {
                if (i == 0) {
                    u_new[i * n + j] = 0.25 * (u_halo[j] + u[i * n + j + 1] + u[i * n + j - 1] + u[(i + 1) * n + j]);
                } else if (i == n - 1) {
                    u_new[i * n + j] = 0.25 * (u[(i - 1) * n + j] + u_halo[j] + u[i * n + j + 1] + u[i * n + j - 1]);
                } else {
                    u_new[i * n + j] = 0.25 * (u[(i - 1) * n + j] + u[(i + 1) * n + j] + u[i * n + j + 1] + u[i * n + j - 1]);
                }
            }
        }

        // update boundaries
        for (long j = 0; j < n; j++) {
            u_new[lo * n + j] = u[lo * n + j];
            u_new[(hi - 1) * n + j] = u[(hi - 1) * n + j];
        }

        double *tmp = u;
        u = u_new;
        u_new = tmp;
    }

    free(u_new);
}

#include "jacobi.h"
#include "reference.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

void jacobi(double *u, long n, int iters) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;

    double *un = (double *) malloc(n * n * sizeof(double));

    for (int iter = 0; iter < iters; iter++) {
        // copy keeps the fixed boundaries
        memcpy(un, u, n * n * sizeof(double));

        // halo exchange
        double *sendbuf = (double *) malloc(n * sizeof(double));
        double *recvbuf = (double *) malloc(n * sizeof(double));

        if (rank > 0) {
            MPI_Sendrecv(u + (lo - 1) * n, n, MPI_DOUBLE, rank - 1, 0,
                          recvbuf, n, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < size - 1) {
            MPI_Sendrecv(u + hi * n, n, MPI_DOUBLE, rank + 1, 0,
                          sendbuf, n, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // update interior cells
        for (long i = lo; i < hi; i++) {
            for (long j = 1; j < n - 1; j++) {
                un[i * n + j] = 0.25 * (u[(i - 1) * n + j] + u[(i + 1) * n + j] + u[i * n + j - 1] + u[i * n + j + 1]);
            }
        }

        // combine results
        MPI_Allgather(un + lo * n, (hi - lo) * n, MPI_DOUBLE,
                       u + lo * n, (hi - lo) * n, MPI_DOUBLE, MPI_COMM_WORLD);

        free(sendbuf);
        free(recvbuf);
    }

    free(un);
}

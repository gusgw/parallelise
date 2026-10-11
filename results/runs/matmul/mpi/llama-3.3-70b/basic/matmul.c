#include "matmul.h"
#include "reference.h"
#include <mpi.h>
#include <stdlib.h>

void matmul(const double *A, const double *B, double *C, long n) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;

    for (long i = lo; i < hi; i++) {
        for (long j = 0; j < n; j++) {
            double s = 0.0;
            for (long k = 0; k < n; k++) {
                s += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = s;
        }
    }

    double *recvbuf = (double *)malloc(n * n * sizeof(double));
    MPI_Allgather(MPI_IN_PLACE, (hi - lo) * n, MPI_DOUBLE, recvbuf, (hi - lo) * n, MPI_DOUBLE, MPI_COMM_WORLD);

    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            C[i * n + j] = recvbuf[i * n + j];
        }
    }

    free(recvbuf);
}

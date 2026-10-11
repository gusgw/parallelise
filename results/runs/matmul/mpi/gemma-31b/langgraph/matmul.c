#include <mpi.h>
#include <stdlib.h>
#include "matmul.h"

void matmul(const double *A, const double *B, double *C, long n) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long lo = (n * rank) / size;
    long hi = (n * (rank + 1)) / size;
    long rows = hi - lo;

    double *local_C = (double *)malloc(rows * n * sizeof(double));
    if (!local_C) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (long i = 0; i < rows; i++) {
        long global_i = lo + i;
        for (long j = 0; j < n; j++) {
            double sum = 0.0;
            for (long k = 0; k < n; k++) {
                sum += A[global_i * n + k] * B[k * n + j];
            }
            local_C[i * n + j] = sum;
        }
    }

    int *recv_counts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    for (int r = 0; r < size; r++) {
        long r_lo = (n * r) / size;
        long r_hi = (n * (r + 1)) / size;
        recv_counts[r] = (int)((r_hi - r_lo) * n);
        displs[r] = (int)(r_lo * n);
    }

    MPI_Allgatherv(local_C, recv_counts[rank], MPI_DOUBLE,
                   C, recv_counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

    free(local_C);
    free(recv_counts);
    free(displs);
}

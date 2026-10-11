#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "jacobi.h"
#include "reference.h"
#include "problem.h"

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

    if (rank == 0) {
        printf("Initial checksum: %f\n", checksum(u, n));
        printf("Row 0 sum: %f\n", u[0]);
        printf("Row 1 sum: %f\n", u[n]);
    }

    jacobi(u, n, iters);

    if (rank == 0) {
        double cs = checksum(u, n);
        printf("Final checksum: %.12f\n", cs);
        printf("Expected: %.12f\n", REF_CHECKSUM);
        printf("Row 0 sum: %f\n", u[0]);
        printf("Row 1 sum: %f\n", u[n]);
        printf("Row 126 sum: %f\n", u[126*n]);
        printf("Row 127 sum: %f\n", u[127*n]);
    }

    free(u);
    MPI_Finalize();
    return 0;
}

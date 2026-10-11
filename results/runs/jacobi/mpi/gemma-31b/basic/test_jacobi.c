#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "jacobi.h"
#include "reference.h"
#include "problem.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long n = REF_N;
    int iters = REF_ITERS;
    double *u = (double *)malloc(n * n * sizeof(double));

    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            u[i * n + j] = u_init(i, j, n);
        }
    }

    jacobi(u, n, iters);

    double cs = checksum(u, n);
    if (rank == 0) {
        printf("Checksum: %f, Expected: %f, Diff: %f\n", cs, REF_CHECKSUM, fabs(cs - REF_CHECKSUM));
        if (fabs(cs - REF_CHECKSUM) > REF_TOL) {
            printf("Test FAILED\n");
            MPI_Finalize();
            return 1;
        }
        printf("Test PASSED\n");
    }

    free(u);
    MPI_Finalize();
    return 0;
}

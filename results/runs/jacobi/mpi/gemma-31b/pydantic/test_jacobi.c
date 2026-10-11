#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "jacobi.h"
#include "reference.h"
#include "problem.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    long n = REF_N;
    int iters = REF_ITERS;
    double *u = (double *)malloc(n * n * sizeof(double));

    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            u[i * n + j] = u_init(i, j, n);
        }
    }

    jacobi(u, n, iters);

    if (rank == 0) {
        double cs = checksum(u, n);
        printf("Checksum: %f\n", cs);
        if (fabs(cs - REF_CHECKSUM) < REF_TOL) {
            printf("PASS\n");
        } else {
            printf("FAIL: expected %f, got %f\n", REF_CHECKSUM, cs);
            MPI_Finalize();
            return 1;
        }
    }

    free(u);
    MPI_Finalize();
    return 0;
}

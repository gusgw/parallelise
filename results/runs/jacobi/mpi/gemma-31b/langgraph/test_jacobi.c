#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "jacobi.h"
#include "reference.h"
#include "problem.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double *u = (double *)malloc(REF_N * REF_N * sizeof(double));
    for (long i = 0; i < REF_N; i++) {
        for (long j = 0; j < REF_N; j++) {
            u[i * REF_N + j] = u_init(i, j, REF_N);
        }
    }

    jacobi(u, REF_N, REF_ITERS);

    if (rank == 0) {
        double cs = checksum(u, REF_N);
        printf("Checksum: %f (Expected: %f, Tol: %f)\n", cs, REF_CHECKSUM, REF_TOL);
        if (fabs(cs - REF_CHECKSUM) > REF_TOL) {
            printf("FAILED\n");
            MPI_Finalize();
            return 1;
        }
        printf("PASSED\n");
    }

    free(u);
    MPI_Finalize();
    return 0;
}

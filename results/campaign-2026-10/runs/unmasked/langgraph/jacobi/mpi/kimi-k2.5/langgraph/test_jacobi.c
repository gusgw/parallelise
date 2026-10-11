#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "jacobi.h"
#include "reference.h"
#include "problem.h"
#include "testing.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    long n = REF_N;
    int iters = (int)REF_ITERS;

    /* Allocate and initialize grid on all ranks */
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

    /* Collective call - all ranks participate */
    jacobi(u, n, iters);

    /* Only rank 0 checks and reports results */
    if (rank == 0) {
        double cs = checksum(u, n);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum");
        int result = TEST_SUMMARY();
        free(u);
        MPI_Finalize();
        return result;
    }

    free(u);
    MPI_Finalize();
    return 0;
}

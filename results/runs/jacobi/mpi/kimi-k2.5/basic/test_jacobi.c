#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "jacobi.h"
#include "reference.h"
#include "problem.h"
#include "testing.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Allocate and initialize grid */
    long n = REF_N;
    int iters = (int)REF_ITERS;
    double *u = malloc(n * n * sizeof(double));

    /* Initialize: top row hot (1.0), everything else cold (0.0) */
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            u[i * n + j] = u_init(i, j, n);
        }
    }

    /* Run Jacobi (collective - all ranks call) */
    jacobi(u, n, iters);

    /* Only rank 0 checks the result */
    int failed = 0;
    if (rank == 0) {
        double cs = checksum(u, n);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum");
        failed = TEST_SUMMARY();
    }

    free(u);
    MPI_Finalize();
    return failed;
}

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "kernel_b.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    long N = REF_N;
    int ITERS = (int)REF_ITERS;
    double *u = (double *)malloc(N * N * sizeof(double));
    if (!u) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    /* Initialize the grid according to the problem definition */
    for (long i = 0; i < N; ++i) {
        for (long j = 0; j < N; ++j) {
            u[i * N + j] = u_init(i, j, N);
        }
    }

    kernel_b(u, N, ITERS);

    if (rank == 0) {
        double chk = checksum(u, N);
        CHECK_CLOSE(chk, REF_CHECKSUM, REF_TOL, "checksum");
        if (tests_failed == 0) {
            printf("PASS\n");
        }
    }

    free(u);
    MPI_Finalize();
    return TEST_SUMMARY();
}

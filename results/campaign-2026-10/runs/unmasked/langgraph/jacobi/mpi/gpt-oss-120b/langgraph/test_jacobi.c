#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "jacobi.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long n = REF_N;
    int iters = (int)REF_ITERS;

    /* Allocate full grid on each rank. */
    double *u = malloc(sizeof(double) * n * n);
    if (!u) {
        fprintf(stderr, "alloc failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (long i = 0; i < n; ++i) {
        for (long j = 0; j < n; ++j) {
            u[i * n + j] = u_init(i, j, n);
        }
    }

    /* Run the kernel. */
    jacobi(u, n, iters);

    if (rank == 0) {
        double chk = checksum(u, n);
        CHECK_CLOSE(chk, REF_CHECKSUM, REF_TOL, "Jacobi checksum");
        if (tests_failed == 0) {
            printf("PASS\n");
        }
    }

    free(u);
    MPI_Finalize();
    return TEST_SUMMARY();
}

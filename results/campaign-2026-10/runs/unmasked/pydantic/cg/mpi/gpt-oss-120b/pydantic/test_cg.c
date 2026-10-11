#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "cg.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long m = REF_M;
    int iters = (int)REF_ITERS;

    /* allocate full vectors */
    double *x = (double *)malloc(sizeof(double) * m * m);
    double *b = (double *)malloc(sizeof(double) * m * m);
    if (!x || !b) {
        if (rank == 0) fprintf(stderr, "allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (long i = 0; i < m * m; ++i) {
        b[i] = b_of(i);
    }

    cgsolve(x, b, m, iters);

    if (rank == 0) {
        double cs = checksum(x, m);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum matches reference");
        if (tests_failed == 0) {
            printf("All checks passed. checksum = %.12g\n", cs);
        }
    }

    free(x);
    free(b);
    MPI_Finalize();
    return TEST_SUMMARY();
}

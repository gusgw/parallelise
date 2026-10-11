#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "cg.h"
#include "reference.h"
#include "problem.h"
#include "testing.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    long m = REF_M;
    int iters = (int)REF_ITERS;
    long n = m * m;

    /* Allocate and fill b (every rank has a full copy) */
    double *b = malloc(n * sizeof(double));
    for (long idx = 0; idx < n; idx++) {
        b[idx] = b_of(idx);
    }

    /* Allocate x for the solution */
    double *x = malloc(n * sizeof(double));

    /* Call the collective kernel */
    cgsolve(x, b, m, iters);

    /* Compute checksum */
    double cs = checksum(x, m);

    /* Only rank 0 checks and prints results */
    if (rank == 0) {
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum");
        int result = TEST_SUMMARY();
        free(b);
        free(x);
        MPI_Finalize();
        return result;
    }

    free(b);
    free(x);
    MPI_Finalize();
    return 0;
}

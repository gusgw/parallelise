#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
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

    /* Allocate full b and x (every rank needs full b, and will receive full x) */
    double *b = (double *)malloc(n * sizeof(double));
    double *x = (double *)malloc(n * sizeof(double));

    /* Fill b deterministically (same on all ranks) */
    for (long idx = 0; idx < n; idx++) {
        b[idx] = b_of(idx);
    }

    /* Call the kernel (collective) */
    cgsolve(x, b, m, iters);

    /* Only rank 0 checks the result */
    if (rank == 0) {
        double cs = checksum(x, m);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum");
    }

    /* Cleanup */
    free(b);
    free(x);

    int failed = 0;
    if (rank == 0) {
        failed = TEST_SUMMARY();
    }

    MPI_Finalize();
    return failed;
}

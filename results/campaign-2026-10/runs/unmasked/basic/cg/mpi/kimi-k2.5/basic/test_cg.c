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

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    long m = REF_M;
    int iters = (int)REF_ITERS;
    long n = m * m;

    /* Allocate and fill b (every rank holds a full copy) */
    double *b = (double *)malloc(n * sizeof(double));
    for (long idx = 0; idx < n; idx++) {
        b[idx] = b_of(idx);
    }

    /* Allocate x for the solution */
    double *x = (double *)malloc(n * sizeof(double));

    /* Call the kernel (collective operation) */
    cgsolve(x, b, m, iters);

    /* Only rank 0 checks the result */
    if (rank == 0) {
        double cs = checksum(x, m);
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

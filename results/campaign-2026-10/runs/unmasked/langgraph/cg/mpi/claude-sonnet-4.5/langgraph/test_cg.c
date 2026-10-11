#include "cg.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    long m = REF_M;
    int iters = REF_ITERS;
    long n = m * m;

    /* Allocate and fill b on all ranks */
    double *b = (double *)malloc(n * sizeof(double));
    for (long idx = 0; idx < n; idx++) {
        b[idx] = b_of(idx);
    }

    /* Allocate x on all ranks */
    double *x = (double *)malloc(n * sizeof(double));

    /* Call the kernel (collective, all ranks must call) */
    cgsolve(x, b, m, iters);

    /* Only rank 0 checks and reports */
    int result = 0;
    if (rank == 0) {
        double cs = checksum(x, m);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum");
        result = TEST_SUMMARY();
    }

    free(b);
    free(x);
    
    MPI_Finalize();
    return result;
}

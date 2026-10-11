#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "cg.h"
#include "reference.h"
#include "problem.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    long m = REF_M;
    int iters = (int)REF_ITERS;
    long n = m * m;

    /* Allocate and fill b (every rank gets a full copy) */
    double *b = (double *)malloc(n * sizeof(double));
    for (long idx = 0; idx < n; idx++) {
        b[idx] = b_of(idx);
    }

    /* Allocate x and initialize to zero */
    double *x = (double *)calloc(n, sizeof(double));

    /* Call the kernel - all ranks participate */
    cgsolve(x, b, m, iters);

    /* Only rank 0 checks the result */
    int passed = 1;
    if (rank == 0) {
        double cs = checksum(x, m);
        double err = fabs(cs - REF_CHECKSUM);
        if (err > REF_TOL) {
            printf("FAIL: checksum = %.12f, expected = %.12f, error = %.12f, tol = %.12f\n",
                   cs, REF_CHECKSUM, err, REF_TOL);
            passed = 0;
        } else {
            printf("PASS: checksum = %.12f, expected = %.12f, error = %.12f\n",
                   cs, REF_CHECKSUM, err);
        }
    }

    free(b);
    free(x);

    MPI_Finalize();
    return passed ? 0 : 1;
}

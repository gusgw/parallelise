#include "cg.h"
#include "problem.h"
#include "reference.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

    /* Call the kernel (collective) */
    cgsolve(x, b, m, iters);

    /* Only rank 0 checks and reports */
    if (rank == 0) {
        double cs = checksum(x, m);
        double err = fabs(cs - REF_CHECKSUM);
        if (err <= REF_TOL) {
            printf("PASS: checksum = %.10f (expected %.10f, err = %.3e)\n",
                   cs, REF_CHECKSUM, err);
        } else {
            printf("FAIL: checksum = %.10f (expected %.10f, err = %.3e > tol %.3e)\n",
                   cs, REF_CHECKSUM, err, REF_TOL);
        }
    }

    free(x);
    free(b);

    MPI_Finalize();
    return 0;
}

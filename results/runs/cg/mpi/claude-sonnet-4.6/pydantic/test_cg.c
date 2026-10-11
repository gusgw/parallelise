/*
 * test_cg.c  --  Unit test for cgsolve().
 *
 * Every rank calls cgsolve() with the same inputs (REF_M x REF_M grid,
 * REF_ITERS iterations, deterministic RHS from b_of()).  Only rank 0
 * checks the checksum against REF_CHECKSUM within REF_TOL and prints
 * the result.  All ranks call MPI_Finalize before exit.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cg.h"
#include "problem.h"
#include "reference.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    long m     = REF_M;
    int  iters = (int)REF_ITERS;
    long n     = m * m;

    /* Allocate and fill the right-hand side (identical on every rank). */
    double *b = (double *)malloc(n * sizeof(double));
    double *x = (double *)malloc(n * sizeof(double));
    if (!b || !x) {
        fprintf(stderr, "rank %d: allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (long t = 0; t < n; t++) b[t] = b_of(t);

    /* Collective solve -- every rank participates. */
    cgsolve(x, b, m, iters);

    /* Only rank 0 checks and reports. */
    int status = 0;
    if (rank == 0) {
        double cs  = checksum(x, m);
        double err = fabs(cs - REF_CHECKSUM);
        if (err <= REF_TOL) {
            printf("PASS  checksum=%.10f  ref=%.10f  err=%.2e  tol=%.2e\n",
                   cs, REF_CHECKSUM, err, REF_TOL);
            status = 0;
        } else {
            printf("FAIL  checksum=%.10f  ref=%.10f  err=%.2e  tol=%.2e\n",
                   cs, REF_CHECKSUM, err, REF_TOL);
            status = 1;
        }
        fflush(stdout);
    }

    free(b);
    free(x);

    MPI_Finalize();
    return status;
}

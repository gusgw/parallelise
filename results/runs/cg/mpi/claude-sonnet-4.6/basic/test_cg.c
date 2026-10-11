/* test_cg.c — unit test for the parallel CG kernel.
 *
 * Every rank calls cgsolve() with the same inputs (identical b, same m and
 * iters).  Only rank 0 checks the result against the oracle in reference.h
 * and prints the test summary.  All ranks call MPI_Init / MPI_Finalize.
 */

#include <mpi.h>
#include <math.h>
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

    long m     = REF_M;
    int  iters = (int)REF_ITERS;
    long n     = m * m;

    /* Build the right-hand side (deterministic, same on every rank) */
    double *b = (double *)malloc(sizeof(double) * n);
    double *x = (double *)malloc(sizeof(double) * n);
    if (!b || !x) {
        fprintf(stderr, "rank %d: allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (long t = 0; t < n; t++) b[t] = b_of(t);

    /* Every rank participates in the collective kernel */
    cgsolve(x, b, m, iters);

    /* Only rank 0 validates */
    if (rank == 0) {
        double cs = checksum(x, m);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL,
                    "CG checksum matches oracle");
        printf("checksum = %.10g  (ref = %.10g  tol = %g)\n",
               cs, (double)REF_CHECKSUM, (double)REF_TOL);
    }

    free(b);
    free(x);

    int result = 0;
    if (rank == 0) result = TEST_SUMMARY();

    MPI_Finalize();
    return result;
}

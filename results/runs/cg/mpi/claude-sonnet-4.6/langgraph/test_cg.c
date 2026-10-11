/* test_cg.c — unit test for the parallel CG kernel.
 *
 * Every rank calls cgsolve() with the same inputs (REF_M x REF_M grid,
 * REF_ITERS iterations, deterministic RHS from b_of()).  Only rank 0
 * checks the checksum against REF_CHECKSUM within REF_TOL and prints the
 * result.  All ranks participate in MPI_Init / MPI_Finalize and call the
 * collective kernel the same number of times so no rank hangs.
 */

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

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

    /* Build the deterministic RHS (every rank holds a full copy). */
    double *b = (double *)malloc((size_t)n * sizeof(double));
    double *x = (double *)malloc((size_t)n * sizeof(double));
    if (!b || !x) {
        fprintf(stderr, "rank %d: allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (long t = 0; t < n; t++) b[t] = b_of(t);

    /* Collective solve — every rank must call this. */
    cgsolve(x, b, m, iters);

    /* Only rank 0 validates and reports. */
    if (rank == 0) {
        double cs = checksum(x, m);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL,
                    "CG checksum matches reference");
        printf("checksum = %.10g  (ref = %.10g  tol = %g)\n",
               cs, (double)REF_CHECKSUM, (double)REF_TOL);
        /* Return value of TEST_SUMMARY drives the process exit code. */
        int rc = TEST_SUMMARY();
        free(b); free(x);
        MPI_Finalize();
        return rc;
    }

    free(b); free(x);
    MPI_Finalize();
    return 0;
}

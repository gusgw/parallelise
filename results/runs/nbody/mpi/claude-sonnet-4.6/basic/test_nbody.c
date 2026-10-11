/* test_nbody.c — unit test for the parallel nbody_accel kernel.
 *
 * Every rank calls nbody_accel the same number of times (so no rank hangs in
 * a collective).  Only rank 0 asserts against the reference values from
 * reference.h and prints the result.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "nbody.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    long n = REF_N;

    /* Allocate and fill positions (every rank holds the full array) */
    double *px = (double *)malloc(sizeof(double) * (size_t)n);
    double *py = (double *)malloc(sizeof(double) * (size_t)n);
    double *ax = (double *)malloc(sizeof(double) * (size_t)n);
    double *ay = (double *)malloc(sizeof(double) * (size_t)n);
    if (!px || !py || !ax || !ay) {
        fprintf(stderr, "rank %d: allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (long i = 0; i < n; i++) {
        px[i] = px_of(i);
        py[i] = py_of(i);
    }

    /* Collective call — every rank participates */
    nbody_accel(px, py, ax, ay, n);

    /* Only rank 0 validates and prints */
    if (rank == 0) {
        double cs = checksum(ax, ay, n);
        /* The tolerance in reference.h is relative; apply it as an absolute
         * tolerance scaled by the reference value, matching the oracle's intent. */
        double tol = REF_TOL * REF_CHECKSUM;
        CHECK_CLOSE(cs, REF_CHECKSUM, tol, "nbody checksum");
        printf("checksum = %.10g  (ref = %.10g, tol = %g)\n",
               cs, (double)REF_CHECKSUM, tol);
    }

    free(px); free(py); free(ax); free(ay);

    int result = 0;
    if (rank == 0) {
        result = TEST_SUMMARY();
    }

    /* Broadcast the exit code so all ranks exit consistently */
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return result;
}

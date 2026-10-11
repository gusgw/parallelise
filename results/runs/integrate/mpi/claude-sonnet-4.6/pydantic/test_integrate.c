/* test_integrate.c — unit tests for the parallel integrate() kernel.
 *
 * Every MPI rank calls integrate() the same number of times (collective),
 * but only rank 0 checks results against reference.h and prints.
 * Exit code: 0 = all tests passed, non-zero = at least one failure.
 */

#include <mpi.h>
#include <math.h>
#include <stdio.h>

#include "integrate.h"
#include "reference.h"
#include "testing.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* ------------------------------------------------------------------ *
     * Test 1: reference case — [0,1] with REF_N_TEST steps must match    *
     *         REF_EXPECTED within REF_TOL.                                *
     * ------------------------------------------------------------------ */
    double result = integrate(REF_A, REF_B, REF_N_TEST);

    /* ------------------------------------------------------------------ *
     * Test 2: small sanity check — [0,1] with 1 000 000 steps; the       *
     *         midpoint rule is O(h²) so error < 1e-10 at this size.      *
     *         We compare against REF_EXPECTED (≈ π) with a loose tol.   *
     * ------------------------------------------------------------------ */
    double result_small = integrate(0.0, 1.0, 1000000L);

    /* ------------------------------------------------------------------ *
     * Test 3: half-interval [0, 0.5] — known value is                    *
     *         ∫₀^0.5 4/(1+x²) dx = 4·arctan(0.5) ≈ 1.8475836...        *
     * ------------------------------------------------------------------ */
    double result_half = integrate(0.0, 0.5, REF_N_TEST);
    double expected_half = 4.0 * atan(0.5);   /* exact: 4·arctan(b) - 4·arctan(a) */

    /* Only rank 0 asserts and prints. */
    if (rank == 0) {
        CHECK_CLOSE(result, REF_EXPECTED, REF_TOL,
                    "integrate([0,1], N=10M) ≈ pi");

        /* result_small should agree with REF_EXPECTED to better than 1e-9 */
        CHECK_CLOSE(result_small, REF_EXPECTED, 1e-9,
                    "integrate([0,1], N=1M) ≈ pi");

        CHECK_CLOSE(result_half, expected_half, REF_TOL,
                    "integrate([0,0.5], N=10M) ≈ 4*atan(0.5)");
    }

    int rc = 0;
    if (rank == 0) {
        rc = TEST_SUMMARY();
    }

    /* Broadcast the exit code so all ranks exit consistently. */
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return rc;
}

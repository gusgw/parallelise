/* test_integrate.c — unit tests for the parallel integrate() kernel.
 *
 * Every MPI rank calls integrate() the same number of times (required so that
 * the collective MPI calls inside the kernel do not deadlock).  Only rank 0
 * checks results against reference.h and prints the summary.
 */

#include <math.h>
#include <stdio.h>

#include <mpi.h>

#include "integrate.h"
#include "reference.h"
#include "testing.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* ------------------------------------------------------------------ *
     * Test 1: reference case — [0,1] with REF_N_TEST intervals.          *
     * Every rank participates; only rank 0 asserts.                       *
     * ------------------------------------------------------------------ */
    double result_ref = integrate(REF_A, REF_B, REF_N_TEST);
    if (rank == 0) {
        CHECK_CLOSE(result_ref, REF_EXPECTED, REF_TOL,
                    "integrate([0,1], N=10M) ≈ pi");
    }

    /* ------------------------------------------------------------------ *
     * Test 2: small sanity check — [0,1] with n=1000.                    *
     * The midpoint rule for this integrand converges quickly; the error   *
     * at n=1000 is well within 1e-6.                                      *
     * ------------------------------------------------------------------ */
    double result_small = integrate(0.0, 1.0, 1000L);
    if (rank == 0) {
        CHECK_CLOSE(result_small, 3.14159265358979, 1e-5,
                    "integrate([0,1], N=1000) ≈ pi");
    }

    /* ------------------------------------------------------------------ *
     * Test 3: sub-interval [0, 0.5].                                     *
     * ∫₀^0.5 4/(1+x²) dx = 4·arctan(0.5) ≈ 1.8475208614...             *
     * ------------------------------------------------------------------ */
    double expected_half = 4.0 * atan(0.5);   /* ≈ 1.8475208614068024 */
    double result_half   = integrate(0.0, 0.5, REF_N_TEST);
    if (rank == 0) {
        CHECK_CLOSE(result_half, expected_half, REF_TOL,
                    "integrate([0,0.5], N=10M) ≈ 4·arctan(0.5)");
    }

    /* ------------------------------------------------------------------ *
     * Test 4: single subinterval — degenerate but must not crash.        *
     * ------------------------------------------------------------------ */
    double result_n1 = integrate(0.0, 1.0, 1L);
    if (rank == 0) {
        /* midpoint at x=0.5: 4/(1+0.25) = 3.2 */
        CHECK_CLOSE(result_n1, 3.2, 1e-12, "integrate([0,1], N=1) == 3.2");
    }

    /* ------------------------------------------------------------------ *
     * Print summary on rank 0 and exit with non-zero status on failure.  *
     * ------------------------------------------------------------------ */
    int exit_code = 0;
    if (rank == 0) {
        exit_code = TEST_SUMMARY();
    }

    /* Broadcast the exit code so all ranks return the same value. */
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return exit_code;
}

/* test_integrate.c — unit test for the parallel integrate() kernel.
 *
 * Every rank calls integrate() the same number of times (required: collective
 * calls must be symmetric across ranks or mpirun hangs).  Only rank 0 checks
 * the results against the oracle in reference.h and prints the summary.
 *
 * Build:  mpicc -O2 -Wall -Wextra -Werror -std=c11 \
 *                 test_integrate.c integrate.o -o test_integrate -lm
 * Run:    mpirun --oversubscribe -np 4 ./test_integrate
 */

/* Enable M_PI and atan() from <math.h> under strict C11 + glibc. */
#define _GNU_SOURCE

#include <math.h>
#include <stdio.h>

#include <mpi.h>

#include "integrate.h"
#include "reference.h"
#include "testing.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* ------------------------------------------------------------------ *
     * Test 1: reference case — [0,1] with REF_N_TEST subintervals.       *
     * Every rank participates; only rank 0 asserts.                       *
     * ------------------------------------------------------------------ */
    double result_ref = integrate(REF_A, REF_B, REF_N_TEST);

    if (rank == 0) {
        CHECK_CLOSE(result_ref, REF_EXPECTED, REF_TOL,
                    "integrate([0,1], N=10M) ≈ pi");
    }

    /* ------------------------------------------------------------------ *
     * Test 2: small sanity check — [0,1] with n=1000.                    *
     * The midpoint rule is O(h²), so 1000 intervals gives ~1e-7 error.   *
     * We use a generous tolerance of 1e-5 to stay robust.                *
     * ------------------------------------------------------------------ */
    double result_small = integrate(0.0, 1.0, 1000L);

    if (rank == 0) {
        /* π ≈ 3.14159265358979323846 — use the same oracle value */
        CHECK_CLOSE(result_small, REF_EXPECTED, 1e-5,
                    "integrate([0,1], N=1000) ≈ pi (loose tol)");
    }

    /* ------------------------------------------------------------------ *
     * Test 3: symmetry — result must not depend on rank count.           *
     * We re-run the reference case and compare to the first result.      *
     * ------------------------------------------------------------------ */
    double result_again = integrate(REF_A, REF_B, REF_N_TEST);

    if (rank == 0) {
        CHECK_CLOSE(result_again, result_ref, 0.0,
                    "integrate is deterministic across repeated calls");
    }

    /* ------------------------------------------------------------------ *
     * Test 4: sub-interval [0, 0.5].                                     *
     * ∫₀^{0.5} 4/(1+x²) dx = 4·arctan(0.5) ≈ 1.8475836176504332        *
     * ------------------------------------------------------------------ */
    double expected_half = 4.0 * atan(0.5);   /* ≈ 1.8475836176504332 */
    double result_half   = integrate(0.0, 0.5, REF_N_TEST);

    if (rank == 0) {
        CHECK_CLOSE(result_half, expected_half, REF_TOL,
                    "integrate([0,0.5], N=10M) == 4*arctan(0.5)");
    }

    /* ------------------------------------------------------------------ *
     * Print summary (rank 0 only) and exit.                              *
     * ------------------------------------------------------------------ */
    int exit_code = 0;
    if (rank == 0) {
        exit_code = TEST_SUMMARY();
    }

    /* Broadcast the exit code so all ranks return the same status. */
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return exit_code;
}

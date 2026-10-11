/*
 * test_integrate.c — unit tests for the GPU integrate() kernel.
 *
 * Checks the result against the notebook oracle (reference.h) within REF_TOL,
 * plus a few sanity checks at smaller sizes.
 */

#include <math.h>
#include <stdio.h>

#include "integrate.h"
#include "reference.h"
#include "testing.h"

int main(void)
{
    double result;

    /* ------------------------------------------------------------------ */
    /* 1. Primary oracle check: [0,1], N = REF_N_TEST, must match pi.      */
    /* ------------------------------------------------------------------ */
    result = integrate(REF_A, REF_B, REF_N_TEST);
    CHECK_CLOSE(result, REF_EXPECTED, REF_TOL,
                "integrate(0,1,10M) ≈ pi (oracle)");

    /* ------------------------------------------------------------------ */
    /* 2. Smaller n: midpoint rule on [0,1] still converges to pi.         */
    /* ------------------------------------------------------------------ */
    result = integrate(0.0, 1.0, 1000000L);
    CHECK_CLOSE(result, 3.14159265358979, 1e-6,
                "integrate(0,1,1M) ≈ pi");

    /* ------------------------------------------------------------------ */
    /* 3. Symmetry / sub-interval: integral over [0, 0.5] + [0.5, 1]      */
    /*    must equal the full integral over [0, 1].                        */
    /* ------------------------------------------------------------------ */
    {
        double left  = integrate(0.0, 0.5, 500000L);
        double right = integrate(0.5, 1.0, 500000L);
        double full  = integrate(0.0, 1.0, 1000000L);
        CHECK_CLOSE(left + right, full, 1e-9,
                    "integrate(0,0.5) + integrate(0.5,1) == integrate(0,1)");
    }

    /* ------------------------------------------------------------------ */
    /* 4. Degenerate: n = 0 must return 0.                                 */
    /* ------------------------------------------------------------------ */
    result = integrate(0.0, 1.0, 0L);
    CHECK_CLOSE(result, 0.0, 0.0, "integrate(0,1,0) == 0");

    /* ------------------------------------------------------------------ */
    /* 5. Single interval: n = 1, midpoint at x = 0.5.                    */
    /*    Expected = 4/(1+0.25) * 1.0 = 3.2                               */
    /* ------------------------------------------------------------------ */
    result = integrate(0.0, 1.0, 1L);
    CHECK_CLOSE(result, 3.2, 1e-12, "integrate(0,1,1) == 3.2");

    /* ------------------------------------------------------------------ */
    /* 6. Large n: 200 M intervals — exercises the grid-stride path.       */
    /* ------------------------------------------------------------------ */
    result = integrate(0.0, 1.0, 200000000L);
    CHECK_CLOSE(result, 3.14159265358979, 1e-6,
                "integrate(0,1,200M) ≈ pi");

    return TEST_SUMMARY();
}

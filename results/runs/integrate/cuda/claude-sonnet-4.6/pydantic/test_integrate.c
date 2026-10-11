/*
 * test_integrate.c — unit tests for the GPU integrate() kernel.
 *
 * Checks the result against the notebook oracle (reference.h) within REF_TOL,
 * plus a few sanity checks on smaller inputs.
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
    /* 1. Oracle check: must match the notebook's expected value.          */
    /* ------------------------------------------------------------------ */
    result = integrate(REF_A, REF_B, REF_N_TEST);
    CHECK_CLOSE(result, REF_EXPECTED, REF_TOL,
                "integrate(0,1,10000000) matches notebook oracle");

    /* ------------------------------------------------------------------ */
    /* 2. Sanity: integral over [0,1] with 1 000 000 steps is close to pi. */
    /* ------------------------------------------------------------------ */
    result = integrate(0.0, 1.0, 1000000L);
    CHECK_CLOSE(result, 3.14159265358979, 1e-5,
                "integrate(0,1,1000000) close to pi");

    /* ------------------------------------------------------------------ */
    /* 3. Symmetry: [0, 0.5] + [0.5, 1] == [0, 1].                        */
    /* ------------------------------------------------------------------ */
    {
        double left  = integrate(0.0, 0.5, 500000L);
        double right = integrate(0.5, 1.0, 500000L);
        double full  = integrate(0.0, 1.0, 1000000L);
        CHECK_CLOSE(left + right, full, 1e-10,
                    "split interval sums to full integral");
    }

    /* ------------------------------------------------------------------ */
    /* 4. Degenerate: n=1 gives a single midpoint evaluation.              */
    /* ------------------------------------------------------------------ */
    {
        /* midpoint of [0,1] is 0.5; 4/(1+0.25)*1.0 = 3.2 */
        result = integrate(0.0, 1.0, 1L);
        CHECK_CLOSE(result, 3.2, 1e-12, "integrate(0,1,1) == 3.2");
    }

    /* ------------------------------------------------------------------ */
    /* 5. Large n: result is still within REF_TOL of pi.                   */
    /* ------------------------------------------------------------------ */
    result = integrate(0.0, 1.0, 50000000L);
    CHECK_CLOSE(result, 3.14159265358979, REF_TOL,
                "integrate(0,1,50000000) within REF_TOL of pi");

    return TEST_SUMMARY();
}

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
    /* 1. Oracle check: [0,1], n = REF_N_TEST, must match REF_EXPECTED     */
    /* ------------------------------------------------------------------ */
    result = integrate(REF_A, REF_B, REF_N_TEST);
    CHECK_CLOSE(result, REF_EXPECTED, REF_TOL,
                "integrate([0,1], N_TEST) ≈ pi");

    /* ------------------------------------------------------------------ */
    /* 2. Sanity: a smaller n still gives a reasonable estimate of pi      */
    /* ------------------------------------------------------------------ */
    result = integrate(0.0, 1.0, 1000000L);
    CHECK_CLOSE(result, 3.14159265358979, 1e-5,
                "integrate([0,1], 1e6) ≈ pi (1e-5)");

    /* ------------------------------------------------------------------ */
    /* 3. Symmetry / sub-interval: [0, 0.5] + [0.5, 1] == [0, 1]          */
    /* ------------------------------------------------------------------ */
    {
        double left  = integrate(0.0, 0.5, 500000L);
        double right = integrate(0.5, 1.0, 500000L);
        double full  = integrate(0.0, 1.0, 1000000L);
        CHECK_CLOSE(left + right, full, 1e-9,
                    "integrate([0,0.5]) + integrate([0.5,1]) == integrate([0,1])");
    }

    /* ------------------------------------------------------------------ */
    /* 4. Edge case: n == 1 (single subinterval, midpoint at 0.5)          */
    /*    4 / (1 + 0.5^2) * 1.0 = 4 / 1.25 = 3.2                          */
    /* ------------------------------------------------------------------ */
    result = integrate(0.0, 1.0, 1L);
    CHECK_CLOSE(result, 3.2, 1e-12, "integrate([0,1], n=1) == 3.2");

    /* ------------------------------------------------------------------ */
    /* 5. Zero-width interval: should return 0                             */
    /* ------------------------------------------------------------------ */
    result = integrate(0.5, 0.5, 1000L);
    CHECK_CLOSE(result, 0.0, 1e-15, "integrate([0.5,0.5], n=1000) == 0");

    return TEST_SUMMARY();
}

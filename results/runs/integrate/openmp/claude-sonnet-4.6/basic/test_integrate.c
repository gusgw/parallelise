#include <math.h>
#include <stdio.h>

#include "integrate.h"
#include "reference.h"
#include "testing.h"

/* pi to enough digits for our tolerance checks */
#define PI_APPROX 3.14159265358979323846

int main(void)
{
    /* ------------------------------------------------------------------ *
     * 1. Tiny smoke-test: known analytic value on a zero-width interval.
     *    integral of 4/(1+x^2) from 0 to 0 must be 0.
     * ------------------------------------------------------------------ */
    double zero = integrate(0.0, 0.0, 1);
    CHECK_CLOSE(zero, 0.0, REF_TOL, "integrate over zero-width interval");

    /* ------------------------------------------------------------------ *
     * 2. Small-n sanity check: 1 subinterval on [0,1].
     *    Midpoint rule with n=1: x=0.5, result = 4/(1+0.25)*1 = 3.2
     * ------------------------------------------------------------------ */
    double one_step = integrate(0.0, 1.0, 1);
    CHECK_CLOSE(one_step, 3.2, 1e-12, "integrate n=1 midpoint");

    /* ------------------------------------------------------------------ *
     * 3. Medium-n convergence: n=1000 should be within 1e-6 of pi.
     * ------------------------------------------------------------------ */
    double medium = integrate(0.0, 1.0, 1000);
    CHECK_CLOSE(medium, PI_APPROX, 1e-6, "integrate n=1000 close to pi");

    /* ------------------------------------------------------------------ *
     * 4. Oracle check: reproduce the notebook's reference value exactly
     *    within REF_TOL (parameters and tolerance from reference.h).
     * ------------------------------------------------------------------ */
    double result = integrate(REF_A, REF_B, REF_N_TEST);
    CHECK_CLOSE(result, REF_EXPECTED, REF_TOL, "integrate oracle (REF_N_TEST)");

    /* ------------------------------------------------------------------ *
     * 5. Linearity: the integral on [0,2] must be positive and finite.
     * ------------------------------------------------------------------ */
    double wider = integrate(0.0, 2.0, REF_N_TEST);
    CHECK(wider > 0.0, "integrate on [0,2] is positive");
    CHECK(isfinite(wider), "integrate on [0,2] is finite");

    return TEST_SUMMARY();
}

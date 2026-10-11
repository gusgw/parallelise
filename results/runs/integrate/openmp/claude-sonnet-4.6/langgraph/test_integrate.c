/*
 * test_integrate.c — unit tests for the midpoint-rule integrate() kernel.
 *
 * Checks:
 *   1. Small-n sanity: integrate(0,1,1) and integrate(0,1,4) are close to π.
 *   2. Reference oracle: integrate(REF_A, REF_B, REF_N_TEST) matches
 *      REF_EXPECTED within REF_TOL (the value frozen from the notebook).
 *   3. Linearity / sub-interval consistency: splitting [0,1] into two halves
 *      and summing gives the same result as integrating the whole interval.
 *   4. Sign / direction: integrate(b, a, n) == -integrate(a, b, n).
 *   5. Zero-width interval returns 0.
 */

#include <math.h>
#include <stdio.h>

#include "integrate.h"
#include "reference.h"
#include "testing.h"

/* Use the oracle value itself as the reference for π so we don't need M_PI. */
#define PI_APPROX 3.14159265358979323846

int main(void)
{
    /* ------------------------------------------------------------------ */
    /* 1. Coarse approximations are in the right ballpark                  */
    /* ------------------------------------------------------------------ */
    double pi_1   = integrate(0.0, 1.0, 1L);
    double pi_4   = integrate(0.0, 1.0, 4L);
    double pi_100 = integrate(0.0, 1.0, 100L);

    CHECK_CLOSE(pi_1,   PI_APPROX, 0.5,  "n=1   within 0.5 of pi");
    CHECK_CLOSE(pi_4,   PI_APPROX, 0.05, "n=4   within 0.05 of pi");
    CHECK_CLOSE(pi_100, PI_APPROX, 1e-4, "n=100 within 1e-4 of pi");

    /* ------------------------------------------------------------------ */
    /* 2. Oracle / reference check (the main gate)                         */
    /* ------------------------------------------------------------------ */
    double pi_ref = integrate(REF_A, REF_B, REF_N_TEST);
    CHECK_CLOSE(pi_ref, REF_EXPECTED, REF_TOL,
                "oracle: integrate(0,1,1e7) matches REF_EXPECTED within REF_TOL");

    /* ------------------------------------------------------------------ */
    /* 3. Result is independent of how we split the interval               */
    /*    integrate(0,1,n) == integrate(0,0.5,n/2) + integrate(0.5,1,n/2) */
    /* ------------------------------------------------------------------ */
    long n_split = 1000000L;
    double whole = integrate(0.0, 1.0,  n_split);
    double left  = integrate(0.0, 0.5,  n_split / 2);
    double right = integrate(0.5, 1.0,  n_split / 2);
    CHECK_CLOSE(left + right, whole, 1e-9,
                "split: left-half + right-half == whole interval");

    /* ------------------------------------------------------------------ */
    /* 4. Reversing the limits negates the result                          */
    /* ------------------------------------------------------------------ */
    long n_rev = 100000L;
    double fwd = integrate(0.0, 1.0, n_rev);
    double rev = integrate(1.0, 0.0, n_rev);
    CHECK_CLOSE(fwd + rev, 0.0, 1e-12,
                "reversed limits: integrate(a,b) + integrate(b,a) == 0");

    /* ------------------------------------------------------------------ */
    /* 5. Zero-width interval returns 0                                    */
    /* ------------------------------------------------------------------ */
    double zero = integrate(0.5, 0.5, 1000L);
    CHECK_CLOSE(zero, 0.0, 1e-15, "zero-width interval returns 0");

    return TEST_SUMMARY();
}

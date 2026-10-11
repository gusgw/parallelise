/* test_integrate.c — unit tests for the midpoint-rule integrate() kernel.
 *
 * Checks:
 *   1. Small known case: ∫₀¹ 4/(1+x²) dx ≈ π with n=1000 (coarse, generous tol).
 *   2. Reference oracle: REF_N_TEST intervals must reproduce REF_EXPECTED within REF_TOL.
 *   3. Additivity: integrating over [0, 0.5] and [0.5, 1] and summing must equal
 *      integrating over [0, 1] with the same total number of subintervals.
 *   4. Convergence: error shrinks as n grows (O(h²) midpoint rule).
 *   5. Non-unit interval: ∫₀² 4/(1+x²) dx = 4·atan(2).
 */

#include <math.h>
#include <stdio.h>

#include "integrate.h"
#include "reference.h"
#include "testing.h"

/* π to 20 significant figures — avoids the M_PI / _GNU_SOURCE portability issue */
#define PI 3.14159265358979323846

int main(void)
{
    /* ------------------------------------------------------------------ */
    /* 1. Coarse smoke test: n=1000, generous tolerance                    */
    /* ------------------------------------------------------------------ */
    double coarse = integrate(0.0, 1.0, 1000L);
    CHECK_CLOSE(coarse, PI, 1e-5, "coarse n=1000 approximates pi");

    /* ------------------------------------------------------------------ */
    /* 2. Oracle / reference check                                         */
    /* ------------------------------------------------------------------ */
    double ref = integrate(REF_A, REF_B, REF_N_TEST);
    CHECK_CLOSE(ref, REF_EXPECTED, REF_TOL, "oracle REF_N_TEST matches REF_EXPECTED");

    /* ------------------------------------------------------------------ */
    /* 3. Additivity: [0,0.5] + [0.5,1] == [0,1] for the same total n     */
    /* ------------------------------------------------------------------ */
    long n_half = 500000L;
    double left  = integrate(0.0, 0.5, n_half);
    double right = integrate(0.5, 1.0, n_half);
    double full  = integrate(0.0, 1.0, 2L * n_half);
    CHECK_CLOSE(left + right, full, 1e-10, "additivity: [0,0.5]+[0.5,1] == [0,1]");

    /* ------------------------------------------------------------------ */
    /* 4. Convergence: error shrinks as n grows (O(h²) midpoint rule)      */
    /* ------------------------------------------------------------------ */
    double e1 = fabs(integrate(0.0, 1.0,   100L) - PI);
    double e2 = fabs(integrate(0.0, 1.0,  1000L) - PI);
    double e3 = fabs(integrate(0.0, 1.0, 10000L) - PI);
    /* Each 10× increase in n should reduce the error by ~100× (O(h²)) */
    CHECK(e2 < e1 / 10.0, "error decreases: n=1000 better than n=100");
    CHECK(e3 < e2 / 10.0, "error decreases: n=10000 better than n=1000");

    /* ------------------------------------------------------------------ */
    /* 5. Non-unit interval: ∫₀² 4/(1+x²) dx = 4·atan(2)                 */
    /* ------------------------------------------------------------------ */
    double expected_0_2 = 4.0 * atan(2.0);   /* exact analytical value */
    double got_0_2 = integrate(0.0, 2.0, REF_N_TEST);
    CHECK_CLOSE(got_0_2, expected_0_2, REF_TOL, "non-unit interval [0,2]");

    return TEST_SUMMARY();
}

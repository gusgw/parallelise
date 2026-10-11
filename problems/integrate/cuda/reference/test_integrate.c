#include <math.h>

#include "integrate.h"
#include "reference.h"
#include "testing.h"

int main(void)
{
    /* the headline oracle: integrate([0,1]) matches pi within tolerance */
    double got = integrate(REF_A, REF_B, REF_N_TEST);
    CHECK_CLOSE(got, REF_EXPECTED, REF_TOL, "integrate([0,1]) approximates pi");

    /* even a coarse grid is in the right ballpark */
    CHECK_CLOSE(integrate(REF_A, REF_B, 1000L), REF_EXPECTED, 1e-3,
                "coarse n is still near pi");

    /* refining the grid must reduce the error (catches a broken step/midpoint) */
    double coarse = integrate(REF_A, REF_B, 100L);
    double fine = integrate(REF_A, REF_B, 100000L);
    CHECK(fabs(fine - REF_EXPECTED) < fabs(coarse - REF_EXPECTED),
          "finer n is more accurate than coarser n");

    return TEST_SUMMARY();
}

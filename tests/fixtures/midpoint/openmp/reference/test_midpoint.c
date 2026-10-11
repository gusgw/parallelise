#include <math.h>

#include "midpoint.h"
#include "reference.h"
#include "testing.h"

int main(void)
{
    /* the headline oracle: the notebook's answer at the reference size */
    double got = midpoint(REF_A, REF_B, REF_N_TEST);
    CHECK_CLOSE(got, REF_EXPECTED, REF_TOL, "midpoint matches the oracle");

    /* a coarse grid is already near the exact value 2 */
    CHECK_CLOSE(midpoint(REF_A, REF_B, 1000L), 2.0, 1e-3, "coarse grid is near 2");

    return TEST_SUMMARY();
}

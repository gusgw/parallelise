#include "integrate.h"
#include "reference.h"
#include "testing.h"

#include <math.h>
#include <stdio.h>

int main(void)
{
    /* Test against the reference values from reference.h */
    double result = integrate(REF_A, REF_B, REF_N_TEST);
    
    CHECK_CLOSE(result, REF_EXPECTED, REF_TOL, "integrate should match reference value");
    
    /* Additional sanity check: result should be close to pi */
    const double pi = acos(-1.0);
    CHECK_CLOSE(result, pi, 1e-6, "integrate should approximate pi");
    
    return TEST_SUMMARY();
}

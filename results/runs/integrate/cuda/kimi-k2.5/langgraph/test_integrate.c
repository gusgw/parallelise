#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "integrate.h"
#include "reference.h"
#include "testing.h"

int main(void)
{
    /* Test against the reference value from the notebook oracle */
    double result = integrate(REF_A, REF_B, REF_N_TEST);
    
    CHECK_CLOSE(result, REF_EXPECTED, REF_TOL, 
                "integrate(0.0, 1.0, 10000000) should match reference value");
    
    return TEST_SUMMARY();
}

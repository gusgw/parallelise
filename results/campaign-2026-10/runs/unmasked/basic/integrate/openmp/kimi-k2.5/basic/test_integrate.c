#include "integrate.h"
#include "reference.h"
#include "testing.h"
#include <math.h>
#include <stdio.h>

int main(void)
{
    double result = integrate(REF_A, REF_B, REF_N_TEST);
    CHECK_CLOSE(result, REF_EXPECTED, REF_TOL, "integrate midpoint rule");
    return TEST_SUMMARY();
}

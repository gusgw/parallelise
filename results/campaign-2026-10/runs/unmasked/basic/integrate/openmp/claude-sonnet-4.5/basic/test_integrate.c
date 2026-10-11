#include <stdio.h>
#include "integrate.h"
#include "reference.h"
#include "testing.h"

int main(void) {
    double result = integrate(REF_A, REF_B, REF_N_TEST);
    
    CHECK_CLOSE(result, REF_EXPECTED, REF_TOL, "integrate(REF_A, REF_B, REF_N_TEST)");
    
    return TEST_SUMMARY();
}

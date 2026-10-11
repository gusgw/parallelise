#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "integrate.h"
#include "reference.h"

int main(void) {
    /* Test the integrate function against reference values */
    double result = integrate(REF_A, REF_B, REF_N_TEST);
    
    double error = fabs(result - REF_EXPECTED);
    
    printf("integrate(%g, %g, %ld) = %.15f\n", REF_A, REF_B, REF_N_TEST, result);
    printf("Expected: %.15f\n", REF_EXPECTED);
    printf("Error: %.15e\n", error);
    printf("Tolerance: %.15e\n", REF_TOL);
    
    if (error <= REF_TOL) {
        printf("TEST PASSED\n");
        return 0;
    } else {
        printf("TEST FAILED: error exceeds tolerance\n");
        return 1;
    }
}

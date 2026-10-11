/* test_integrate.c - Unit test for integrate function
 * Checks that the CUDA implementation matches the reference oracle
 */

#include <stdio.h>
#include "integrate.h"
#include "reference.h"
#include "testing.h"

int main(void) {
    /* Test the integration against the reference oracle */
    double result = integrate(REF_A, REF_B, REF_N_TEST);
    
    printf("integrate(%g, %g, %ld) = %.12g\n", REF_A, REF_B, REF_N_TEST, result);
    printf("expected = %.12g\n", REF_EXPECTED);
    printf("tol = %g\n", REF_TOL);
    
    CHECK_CLOSE(result, REF_EXPECTED, REF_TOL, "integrate result matches oracle");
    
    return TEST_SUMMARY();
}

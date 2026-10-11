#include "integrate.h"
#include "reference.h"
#include <math.h>
#include <stdio.h>
int main() {
    double result = integrate(REF_A, REF_B, REF_N_TEST);
    if (fabs(result - REF_EXPECTED) > REF_TOL) {
        printf("Test failed: expected %f, got %f\n", REF_EXPECTED, result);
        return 1;
    }
    printf("Test passed\n");
    return 0;
}

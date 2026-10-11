#include "kernel_a.h"
#include "reference.h"
#include <math.h>
#include <stdio.h>

int main() {
    double result = kernel_a(REF_A, REF_B, REF_N_TEST);
    if (fabs(result - REF_EXPECTED) > REF_TOL) {
        printf("Test failed: result = %f, expected = %f\n", result, REF_EXPECTED);
        return 1;
    }
    printf("Test passed\n");
    return 0;
}

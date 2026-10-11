#include <stdio.h>
#include <math.h>
#include "integrate.h"
#include "reference.h"

int main() {
    double result = integrate(REF_A, REF_B, REF_N_TEST);
    double diff = fabs(result - REF_EXPECTED);
    printf("Result: %.15f\n", result);
    printf("Expected: %.15f\n", REF_EXPECTED);
    printf("Diff: %.15f\n", diff);

    if (diff <= REF_TOL) {
        printf("PASS\n");
        return 0;
    } else {
        printf("FAIL\n");
        return 1;
    }
}

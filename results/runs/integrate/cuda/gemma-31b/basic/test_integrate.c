#include <stdio.h>
#include <math.h>
#include "integrate.h"
#include "reference.h"

int main() {
    double result = integrate(REF_A, REF_B, REF_N_TEST);
    double diff = fabs(result - REF_EXPECTED);
    printf("Result: %.15f, Expected: %.15f, Diff: %.15f\n", result, REF_EXPECTED, diff);
    if (diff <= REF_TOL) {
        printf("PASS\n");
        return 0;
    } else {
        printf("FAIL\n");
        return 1;
    }
}

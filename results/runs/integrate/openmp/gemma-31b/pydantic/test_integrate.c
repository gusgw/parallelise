#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "integrate.h"
#include "reference.h"

int main() {
    double result = integrate(REF_A, REF_B, REF_N_TEST);
    double error = fabs(result - REF_EXPECTED);

    printf("Result:   %.15f\n", result);
    printf("Expected: %.15f\n", REF_EXPECTED);
    printf("Error:    %.15f\n", error);

    if (error <= REF_TOL) {
        printf("PASS\n");
        return 0;
    } else {
        printf("FAIL\n");
        return 1;
    }
}

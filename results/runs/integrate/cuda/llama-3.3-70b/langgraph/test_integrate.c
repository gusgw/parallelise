// test_integrate.c: unit test for integrate.cu
#include <stdio.h>
#include <math.h>
#include "reference.h"
#include "integrate.h"

int main() {
    double approx = integrate(REF_A, REF_B, REF_N_TEST);
    printf("approx = %f\n", approx);
    double error = fabs(approx - REF_EXPECTED);
    printf("error = %f\n", error);
    if (error <= REF_TOL) {
        printf("PASSED\n");
        return 0;
    } else {
        printf("FAILED\n");
        return 1;
    }
}

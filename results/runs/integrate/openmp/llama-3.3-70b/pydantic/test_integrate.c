#include "integrate.h"
#include "reference.h"
#include <math.h>
#include <stdio.h>

int main() {
    double approx = integrate(REF_A, REF_B, REF_N_TEST);
    double error = fabs(approx - REF_EXPECTED);
    printf("approx = %f  error = %f\n", approx, error);
    if (error <= REF_TOL) {
        printf("PASS\n");
        return 0;
    } else {
        printf("FAIL\n");
        return 1;
    }
}

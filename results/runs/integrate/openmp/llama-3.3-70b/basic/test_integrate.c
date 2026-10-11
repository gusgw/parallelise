#include "integrate.h"
#include "reference.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    double approx = integrate(REF_A, REF_B, REF_N_TEST);
    double error = fabs(approx - REF_EXPECTED);
    if (error > REF_TOL) {
        printf("approx = %f  error = %f\n", approx, error);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

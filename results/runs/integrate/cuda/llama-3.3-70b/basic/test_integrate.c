#include "integrate.h"
#include "reference.h"
#include <math.h>
#include <stdio.h>

int main() {
    double result = integrate(REF_A, REF_B, REF_N_TEST);
    if (fabs(result - REF_EXPECTED) <= REF_TOL) {
        printf("PASS\n");
        return 0;
    } else {
        printf("FAIL\n");
        return 1;
    }
}
